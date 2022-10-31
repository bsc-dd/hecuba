import random
import string
import uuid
from collections import namedtuple

from hecuba.hfetch import Hcache

from . import config, log
from .IStorage import IStorage
from .storageiter import NamedItemsIterator


class QbeastMeta(object):
    def __init__(self, mem_filter, from_point, to_point, precision):
        self.precision = precision
        self.from_point = from_point
        self.to_point = to_point
        self.mem_filter = mem_filter


config.cluster.register_user_type('hecuba', 'q_meta', QbeastMeta)


class QbeastIterator(IStorage):
    """
    Object used to access data from workers.
    """

    args_names = ['primary_keys', 'columns', 'indexed_on', 'name', 'qbeast_meta', 'qbeast_random',
                  'storage_id', 'tokens', 'class_name', 'built_remotely']
    _building_args = namedtuple('QbeastArgs', args_names)
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba.istorage'
                                                  '(primary_keys, columns, indexed_on, name, qbeast_meta,'
                                                  ' qbeast_random, storage_id, tokens, class_name)'
                                                  'VALUES (?,?,?,?,?,?,?,?,?)')
    _prepared_set_qbeast_meta = config.session.prepare('INSERT INTO hecuba.istorage (storage_id, qbeast_meta) '
                                                       'VALUES (?,?)')

    @staticmethod
    def _store_meta(storage_args):
        log.debug("QbeastIterator: storing metas %s", '')

        try:
            config.session.execute(QbeastIterator._prepared_store_meta,
                                   [storage_args.primary_keys,
                                    storage_args.columns,
                                    storage_args.indexed_on,
                                    storage_args.name,
                                    storage_args.qbeast_meta,
                                    storage_args.qbeast_random,
                                    storage_args.storage_id,
                                    storage_args.tokens,
                                    storage_args.class_name])
        except Exception as ex:
            log.error("Error creating the StorageDictIx metadata: %s %s", storage_args, ex)
            raise ex

    def __init__(self, primary_keys, columns, indexed_on, name, qbeast_meta=None, qbeast_random=None,
                 storage_id=None, tokens=None, **kwargs):
        """
        Creates a new block.
        Args:
            primary_keys (list(tuple)): a list of (key,type) primary keys (primary + clustering).
            columns (list(tuple)): a list of (key,type) columns
            indexed_on (list(str)): a list of the names of the indexed columns
            name (string): keyspace.table of the Cassandra collection
            qbeast_random (str): qbeast random string, when selecting in different nodes this must have the same value
            storage_id (uuid): the storage id identifier
            tokens (list): list of tokens
        """
        super().__init__((), name=name, storage_id=storage_id, **kwargs)

        log.debug("CREATED QbeastIterator(%s,%s,%s,%s)", storage_id, tokens, )

        self._qbeast_meta = qbeast_meta
        self._primary_keys = primary_keys
        self._columns = columns
        self._indexed_on = indexed_on

        if qbeast_random is None:
            self._qbeast_random = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(5))
        else:
            self._qbeast_random = qbeast_random

        class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)

        self._primary_keys = [{"type": key[1], "name": key[0]} if isinstance(key, tuple) else key
                              for key in self._primary_keys]
        self._columns = [{"type": col[1], "name": col[0]} if isinstance(col, tuple) else col
                         for col in self._columns]

        key_names = [col["name"] for col in self._primary_keys]
        column_names = [col["name"] for col in self._columns]
        if len(key_names) > 1:
            self._key_builder = namedtuple('row', key_names)
        else:
            self._key_builder = None
        if len(column_names) > 1:
            self._column_builder = namedtuple('row', column_names)
        else:
            self._column_builder = None

        self._k_size = len(primary_keys)

        build_keys = [(key["name"], key["type"]) for key in self._primary_keys]
        build_columns = [(col["name"], col["type"]) for col in self._columns]

        self._build_args = self._building_args(
            build_keys,
            build_columns,
            self._indexed_on,
            self._ksp + "." + self._table,
            self._qbeast_meta,
            self._qbeast_random,
            self.storage_id,
            self._tokens,
            class_name,
            self._built_remotely)

        if name or storage_id:
            self.make_persistent(name)

    def make_persistent(self, name):
        # Update local QbeastIterator metadata
        super().make_persistent(name)
        self._build_args = self._build_args._replace(storage_id=self.storage_id, name=self._ksp + "." + self._table,
                                                     tokens=self._tokens)

        self._setup_hcache()

        QbeastIterator._store_meta(self._build_args)

    def _setup_hcache(self):
        key_names = [key["name"] for key in self._primary_keys]
        persistent_values = [{"name": col["name"]} for col in self._columns]

        if self._tokens is None:
            raise RuntimeError("Tokens for object {} are null".format(self._get_name()))

        self._hcache_params = (self._ksp, self._table,
                               self.storage_id,
                               self._tokens, key_names, persistent_values,
                               {'cache_size': config.max_cache_size,
                                'writer_par': config.write_callbacks_number,
                                'writer_buffer': config.write_buffer_size,
                                'timestamped_writes': config.timestamped_writes})
        log.debug("HCACHE params %s", self._hcache_params)
        self._hcache = Hcache(*self._hcache_params)

    def _set_qbeast_meta(self, qbeast_meta):
        self._qbeast_meta = qbeast_meta
        self._build_args = self._build_args._replace(qbeast_meta=qbeast_meta)
        config.session.execute(QbeastIterator._prepared_set_qbeast_meta, [self.storage_id, qbeast_meta])

    def __len__(self):
        return len([row for row in self.__iter__()])

    def __iter__(self):
        if hasattr(self, "_qbeast_meta") and self._qbeast_meta is not None:
            conditions = ""
            for index, (from_p, to_p) in enumerate(zip(self._qbeast_meta.from_point, self._qbeast_meta.to_point)):
                conditions += "{0} > {1} AND {0} < {2} AND ".format(self._indexed_on[index], from_p, to_p)

            conditions = conditions[:-5] + self._qbeast_meta.mem_filter

            conditions += " AND expr(%s_idx, 'precision=%s:%s') ALLOW FILTERING" \
                          % (self._table, self._qbeast_meta.precision, self._qbeast_random)

            hiter = self._hcache.iteritems({'custom_select': conditions, 'prefetch_size': config.prefetch_size})
        else:
            hiter = self._hcache.iteritems(config.prefetch_size)

        return NamedItemsIterator(self._key_builder, self._column_builder, self._k_size, hiter, self)
