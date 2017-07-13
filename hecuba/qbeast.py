import uuid
from collections import namedtuple
from exceptions import ValueError
from struct import *

from hecuba import config
from hecuba import log
from hecuba.IStorage import IStorage
from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket
from thrift.transport import TTransport
from hecuba.qthrift import QbeastMaster
from hecuba.qthrift import QbeastWorker
from hecuba.qthrift.ttypes import BasicTypes
from hecuba.qthrift.ttypes import FilteringArea


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
    args_names = ['primary_keys', 'columns', 'name', 'qbeast_meta', 'qbeast_id',
                  'entry_point', 'storage_id', 'tokens', 'class_name']
    _building_args = namedtuple('StorageDictArgs', args_names)
    _prepared_store_meta = config.session.prepare(
        'INSERT INTO hecuba.istorage'
        '(primary_keys, columns, name, qbeast_meta,'
        ' qbeast_id, entry_point, storage_id, tokens, class_name)'
        'VALUES (?,?,?,?,?,?,?,?,?)')
    _prepared_set_qbeast_id = config.session.prepare(
        'INSERT INTO hecuba.istorage (storage_id,qbeast_id)VALUES (?,?)')
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
                              result.name,
                              result.qbeast_meta,
                              result.qbeast_id,
                              result.entry_point,
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
                                    storage_args.entry_point,
                                    storage_args.storage_id,
                                    storage_args.tokens,
                                    storage_args.class_name])
        except Exception as ex:
            log.error("Error creating the StorageDictIx metadata: %s %s", storage_args, ex)
            # raise ex

    def __init__(self, primary_keys, columns, name, qbeast_meta,
                 qbeast_id=None,
                 entry_point=None, storage_id=None, tokens=None):
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
        self._selects = map(lambda a: a[0], primary_keys + columns)
        key_namedtuple = namedtuple("key", map(lambda a: a[0], primary_keys))
        value_namedtuple = namedtuple("value", map(lambda a: a[0], columns))
        div = len(primary_keys)
        self._row_builder = lambda a: self._row_namedtuple(key_namedtuple(*a[:div]), value_namedtuple(*a[div:]))
        (self._ksp, self._table) = self._extract_ks_tab(name)
        self._qbeast_meta = qbeast_meta
        self._qbeast_id = qbeast_id
        self._entry_point = entry_point
        if tokens is None:
            log.info('using all tokens')
            tokens = map(lambda a: a.value, config.cluster.metadata.token_map.ring)
            self._tokens = IStorage._discrete_token_ranges(tokens)
        else:
            self._tokens = tokens

        class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)

        # primary_keys columns name tokens
        # indexed_args nonindexed_args value_list
        # mem_filter port storage_id class_name
        if storage_id is None:
            self._storage_id = uuid.uuid4()
        else:
            self._storage_id = storage_id
        self._build_args = self._building_args(
            primary_keys,
            columns,
            name,
            qbeast_meta,
            qbeast_id,
            entry_point,
            self._storage_id,
            self._tokens,
            class_name)
        if storage_id is None:
            self._store_meta(self._build_args)

    def split(self):
        """
        Initializes the iterator, and saves the information about the token ranges of each block
        Args:
            my_dict (PersistentDict): Hecuba PersistentDict
        """
        splits = [s for s in IStorage.split(self)]

        if type(config.qbeast_entry_node) == list:
            qbeast_node = config.qbeast_entry_node[0]
        else:
            qbeast_node = config.qbeast_entry_node

        transport = TSocket.TSocket(qbeast_node, config.qbeast_master_port)

        # Buffering is critical. Raw sockets are very slow
        transport = TTransport.TFramedTransport(transport)

        # Wrap in a protocol
        protocol = TBinaryProtocol.TBinaryProtocol(transport)

        # Create a client to use the protocol encoder
        client = QbeastMaster.Client(protocol)

        # Connect!
        transport.open()

        area = FilteringArea(fromPoint=self._qbeast_meta.from_point,
                             toPoint=self._qbeast_meta.to_point)
        uuids = map(lambda x: str(x._storage_id), splits)

        log.info("calling initQuery (%s, %s, %s, precision=%f ,area=%s, uuids=%s, max_results=%f",
                 self._selects,
                 self._ksp,
                 self._table,
                 self._qbeast_meta.precision,
                 area, uuids,
                 config.qbeast_max_results)

        self._qbeast_id = uuid.UUID(client.initQuery(self._selects,
                                                     self._ksp + '_qbeast',
                                                     self._table + '_' + self._table + '_idx_d8tree',
                                                     area,
                                                     self._qbeast_meta.precision,
                                                     config.qbeast_max_results,
                                                     uuids))
        transport.close()

        for i in splits:
            i._set_qbeast_id(self._qbeast_id)

        return iter(splits)

    def _set_qbeast_id(self, qbeast_id):
        self._qbeast_id = qbeast_id
        self._build_args = self._build_args._replace(qbeast_id=qbeast_id)
        config.session.execute(QbeastIterator._prepared_set_qbeast_id, [qbeast_id])

    def getID(self):
        return str(self._storage_id)

    def __eq__(self, other):
        return self._storage_id == other._storage_id and \
               self._tokens == other.token_ranges \
               and self._table == other.table_name and self._ksp == other.keyspace

    def __len__(self):
        i = 0
        for _ in self:
            i += 1
        return i

    def __iter__(self):
        if self._storage_id is None or self._qbeast_id is None:
            '''
             In this case, we are doing a query without splitting the object,
             and thus we have to initialize the query for only this iterator
            '''
            if type(config.qbeast_entry_node) == list:
                qbeast_node = config.qbeast_entry_node[0]
            else:
                qbeast_node = config.qbeast_entry_node

            transport = TSocket.TSocket(qbeast_node, config.qbeast_master_port)

            # Buffering is critical. Raw sockets are very slow
            transport = TTransport.TFramedTransport(transport)

            # Wrap in a protocol
            protocol = TBinaryProtocol.TBinaryProtocol(transport)

            # Create a client to use the protocol encoder
            client = QbeastMaster.Client(protocol)

            # Connect!
            transport.open()

            area = FilteringArea(fromPoint=self._qbeast_meta.from_point,
                                 toPoint=self._qbeast_meta.to_point)

            if self._storage_id is None:
                self._storage_id = uuid.uuid4()
                self._build_args = self._build_args._replace(storage_id=self._storage_id)
                self._store_meta(self._build_args)
            uuids = [str(self._storage_id)]

            log.info("calling initQuery (%s, %s, %s, precision=%f ,area=%s, uuids=%s, max_results=%f",
                     self._selects,
                     self._ksp,
                     self._table,
                     self._qbeast_meta.precision,
                     area, uuids,
                     config.qbeast_max_results)

            self._qbeast_id = client.initQuery(self._selects,
                                               self._ksp + '_qbeast',
                                               self._table + '_' + self._table + '_idx_d8tree',
                                               area,
                                               self._qbeast_meta.precision,
                                               config.qbeast_max_results,
                                               uuids)
            self._store_meta(self._build_args)
            transport.close()

        return IndexedIterValue(self._storage_id, self._entry_point, self._row_builder)


class Row:
    def __init__(self, metadata, row):
        self._list = []
        for key, value in row.iteritems():
            v = Row.deserialize_type(metadata[key].type, value)
            cname = metadata[key].columnName
            setattr(self, cname, v)
            if cname is not 'rand':
                self._list.append(v)

    def __getitem__(self, key):
        return self._list[key]

    def __str__(self):
        return '(%s)' % (','.join(map(str, self._list)))

    @staticmethod
    def deserialize_type(type, data):
        """
        :param data: bytes
        :type type: BasicTypes
        """
        if type == BasicTypes.BIGINT:
            val, = unpack('!q', data)
            return val
        if type == BasicTypes.BLOB:
            size, = unpack('!i', data[0:4])
            return data[4:size + 4]
        if type == BasicTypes.BOOLEAN:
            val, = unpack('!b', data)
            if val == 0:
                return False
            else:
                return True
        if type == BasicTypes.DOUBLE:
            val, = unpack('!d', data)
            return val
        if type == BasicTypes.FLOAT:
            val, = unpack('!f', data)
            return val
        if type == BasicTypes.INT:
            val, = unpack('!i', data)
            return val
        if type == BasicTypes.TEXT:
            bin = Row.deserialize_type(BasicTypes.BLOB, data)
            return bin.decode('utf8')
        raise ValueError


class IndexedIterValue(object):
    def __iter__(self):
        return self

    def __init__(self, storage_id, host, row_builder):
        '''

        Args:
            storage_dict: type IxBlock
        '''

        self._row_builder = row_builder
        self._storage_id = storage_id
        workerT = TTransport.TFramedTransport(TSocket.TSocket(host, config.qbeast_worker_port))

        # Buffering is critical. Raw sockets are very slow

        # Wrap in a protocol
        pw = TBinaryProtocol.TBinaryProtocol(workerT)

        workerT.open()
        # Create a client to use the protocol encoder
        self._worker = QbeastWorker.Client(pw)
        self._tmp_iter = None
        self._has_more = True

    def next(self):

        if self._tmp_iter is not None:
            try:
                return self._row_builder(self._tmp_iter.next())
            except StopIteration:
                if self._has_more is False:
                    raise StopIteration

        # TODO remove hardcoded parameters
        result = self._worker.get(str(self._storage_id), 1000, 10000)
        self._has_more = result.hasMore
        self._tmp_iter = IndexedIterValue.deserialize(result.metadata, result.data)
        return self._row_builder(self._tmp_iter.next())

    @staticmethod
    def deserialize(metadata, rows):
        # type: (object, object) -> object
        """
        :type metadata: map
        :type rows: list[map]
        """
        for row in rows:
            yield Row(metadata, row)
