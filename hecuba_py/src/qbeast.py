import random
import string
import uuid
from collections import namedtuple

from hecuba import config, log
from hecuba.tools import NamedItemsIterator
from hfetch import Hcache

from IStorage import IStorage


class QbeastMeta(object):
    def __init__(self, mem_filter, from_point, to_point, precision):
        self.precision = precision
        self.to_point = to_point
        self.from_point = from_point
        self.mem_filter = mem_filter


config.cluster.register_user_type('hecuba', 'q_meta', QbeastMeta)


class QbeastIterator(IStorage):
    """
    Object used to access data from workers.
    """
    args_names = ['primary_keys', 'columns', 'indexed_on', 'name', 'qbeast_meta', 'qbeast_id',
                  'storage_id', 'tokens', 'class_name']
    _building_args = namedtuple('StorageDictArgs', args_names)
    _prepared_store_meta = config.session.prepare(
        'INSERT INTO hecuba.istorage'
        '(primary_keys, columns, name, qbeast_meta,'
        ' qbeast_id, storage_id, tokens, class_name)'
        'VALUES (?,?,?,?,?,?,?,?)')
    _prepared_set_qbeast_id = config.session.prepare(
        'INSERT INTO hecuba.istorage (storage_id, qbeast_id)VALUES (?,?)')
    _prepared_set_qbeast_meta = config.session.prepare(
        'INSERT INTO hecuba.istorage (storage_id, qbeast_meta)VALUES (?,?)')
    _row_namedtuple = namedtuple("row", "key,value")

    @staticmethod
    def build_remotely(result):
        """
        Launches the Block.__init__ from the api.getByID
        Args:
            result: a namedtuple with all  the information needed to create again the block
        """
        log.debug("Building Storage dict with %s", result)

        return QbeastIterator(result.primary_keys,
                              result.columns,
                              result.indexed_on,
                              result.name,
                              result.qbeast_meta,
                              result.qbeast_id,
                              result.storage_id,
                              result.tokens
                              )

    @staticmethod
    def _store_meta(storage_args):
        log.debug("QbeastIterator: storing metas %s", '')

        try:
            config.session.execute(QbeastIterator._prepared_store_meta,
                                   [storage_args.primary_keys,
                                    storage_args.columns,
                                    storage_args.name,
                                    storage_args.qbeast_meta,
                                    storage_args.qbeast_id,
                                    storage_args.storage_id,
                                    storage_args.tokens,
                                    storage_args.class_name])
        except Exception as ex:
            log.error("Error creating the StorageDictIx metadata: %s %s", storage_args, ex)
            raise ex

    def __init__(self, primary_keys, columns, indexed_args, name, qbeast_meta=None, qbeast_id=None,
                 storage_id=None, tokens=None):
        """
        Creates a new block.
        Args:
            table_name (string): the name of the collection/table
            keyspace_name (string): name of the Cassandra keyspace.
            primary_keys (list(tuple)): a list of (key,type) primary keys (primary + clustering).
            columns (list(tuple)): a list of (key,type) columns
            tokens (list): list of tokens
            storage_id (uuid): the storage id identifier
        """
        log.debug("CREATED QbeastIterator(%s,%s,%s,%s)", storage_id, tokens, )
        (self._ksp, self._table) = self._extract_ks_tab(name)
        self._primary_keys = primary_keys
        self._columns = columns
        self._indexed_args = indexed_args
        self._qbeast_meta = qbeast_meta
        self._qbeast_id = qbeast_id
        if tokens is None:
            log.info('using all tokens')
            tokens = map(lambda a: a.value, config.cluster.metadata.token_map.ring)
            self._tokens = IStorage._discrete_token_ranges(tokens)
        else:
            self._tokens = tokens

        class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)

        key_names = [pkname for (pkname, dt) in primary_keys]
        column_names = [colname for (colname, dt) in columns]
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
            save = True
        else:
            self._storage_id = storage_id
            save = False
        self._build_args = self._building_args(
            primary_keys,
            columns,
            indexed_args,
            self._ksp + "." + self._table,
            qbeast_meta,
            qbeast_id,
            self._storage_id,
            self._tokens,
            class_name)
        if save:
            self._store_meta(self._build_args)

        persistent_columns = [{"name": col[0], "type": col[1]} for col in columns]
        self._hcache_params = (self._ksp, self._table,
                               self._storage_id,
                               self._tokens, key_names, persistent_columns,
                               {'cache_size': config.max_cache_size,
                                'writer_par': config.write_callbacks_number,
                                'writer_buffer': config.write_buffer_size})
        log.debug("HCACHE params %s", self._hcache_params)
        self._hcache = Hcache(*self._hcache_params)

    def _set_qbeast_id(self, qbeast_id):
        self._qbeast_id = qbeast_id
        self._build_args = self._build_args._replace(qbeast_id=qbeast_id)
        config.session.execute(QbeastIterator._prepared_set_qbeast_id, [self._storage_id, qbeast_id])

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
                conditions += "{0} > {1} AND {0} < {2} AND ".format(self._indexed_args[index], from_p, to_p)

            conditions = conditions[:-5] + self._qbeast_meta.mem_filter

            random_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
            conditions += " AND expr(%s_idx, 'precision=%s:%s') ALLOW FILTERING" \
                          % (self._table, self._qbeast_meta.precision, random_string)

            hiter = self._hcache.iteritems({'custom_select': conditions, 'prefetch_size': config.prefetch_size})
        else:
            hiter = self._hcache.iteritems(config.prefetch_size)

        iterator = NamedItemsIterator(self._key_builder,
                                      self._column_builder,
                                      self._k_size,
                                      hiter,
                                      self)

        return iterator
