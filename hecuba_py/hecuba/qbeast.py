import random
import string
import uuid
from collections import namedtuple

from hecuba import config, log
from hecuba.tools import NamedItemsIterator
from hfetch import Hcache

from IStorage import IStorage, _discrete_token_ranges, _extract_ks_tab


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
    _prepared_store_meta = config.session.prepare(
        'INSERT INTO hecuba.istorage'
        '(primary_keys, columns, indexed_on, name, qbeast_meta,'
        ' qbeast_random, storage_id, tokens, class_name)'
        'VALUES (?,?,?,?,?,?,?,?,?)')
    _prepared_set_qbeast_meta = config.session.prepare(
        'INSERT INTO hecuba.istorage (storage_id, qbeast_meta)VALUES (?,?)')
    _row_namedtuple = namedtuple("row", "key,value")

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
                 storage_id=None, tokens=None, built_remotely=False):
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
        log.debug("CREATED QbeastIterator(%s,%s,%s,%s)", storage_id, tokens, )
        (self._ksp, self._table) = _extract_ks_tab(name)
        self._indexed_on = indexed_on
        self._qbeast_meta = qbeast_meta
        if qbeast_random is None:
            self._qbeast_random = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(5))
        else:
            self._qbeast_random = qbeast_random
        if tokens is None:
            log.info('using all tokens')
            tokens = map(lambda a: a.value, config.cluster.metadata.token_map.ring)
            self._tokens = _discrete_token_ranges(tokens)
        else:
            self._tokens = tokens

        class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)

        key_names = [col[0] if isinstance(col, tuple) else col["name"] for col in primary_keys]
        column_names = [col[0] if isinstance(col, tuple) else col["name"] for col in columns]
        if len(key_names) > 1:
            self._key_builder = namedtuple('row', key_names)
        else:
            self._key_builder = None
        if len(column_names) > 1:
            self._column_builder = namedtuple('row', column_names)
        else:
            self._column_builder = None

        self._k_size = len(primary_keys)

        if storage_id is None:
            self._storage_id = uuid.uuid4()
        else:
            self._storage_id = storage_id

        build_keys = [(key["name"], key["type"]) if isinstance(key, dict) else key for key in primary_keys]
        build_columns = [(col["name"], col["type"]) if isinstance(col, dict) else col for col in columns]

        self._build_args = self._building_args(
            build_keys,
            build_columns,
            self._indexed_on,
            self._ksp + "." + self._table,
            self._qbeast_meta,
            self._qbeast_random,
            self._storage_id,
            self._tokens,
            class_name,
            built_remotely)

        persistent_columns = [{"name": col[0], "type": col[1]} if isinstance(col, tuple) else col for col in columns]

        self._hcache_params = (self._ksp, self._table,
                               self._storage_id,
                               self._tokens, key_names, persistent_columns,
                               {'cache_size': config.max_cache_size,
                                'writer_par': config.write_callbacks_number,
                                'writer_buffer': config.write_buffer_size,
                                'timestamped_writes': config.timestamped_writes})
        log.debug("HCACHE params %s", self._hcache_params)
        self._hcache = Hcache(*self._hcache_params)
      
        if not built_remotely:
            self._store_meta(self._build_args)

    def _set_qbeast_meta(self, qbeast_meta):
        self._qbeast_meta = qbeast_meta
        self._build_args = self._build_args._replace(qbeast_meta=qbeast_meta)
        config.session.execute(QbeastIterator._prepared_set_qbeast_meta, [self._storage_id, qbeast_meta])

    def __eq__(self, other):
        return self._storage_id == other._storage_id and \
               self._tokens == other.token_ranges \
               and self._table == other.table_name and self._ksp == other.keyspace

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

        iterator = NamedItemsIterator(self._key_builder,
                                      self._column_builder,
                                      self._k_size,
                                      hiter,
                                      self)

        return iterator
