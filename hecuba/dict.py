# author: G. Alomar
import logging
import time

from cassandra.query import BatchStatement
from cassandra.query import BatchType

from hecuba import config
from hecuba.cache import PersistentDictCache
from hecuba.iter import KeyIter
from hecuba.prefetchmanager import PrefetchManager


class PersistentDict(dict):
    """
    This class servers as a proxy to Cassandra. You can access to this class as it was a normal python
    dictionary while under the hood it queries Cassandra.
    """

    prepQueriesDict = {}
    createdColumnfamiliesDict = {}
    types = {}
    _insert_data = ''
    prefetchManager = ""
    blockKeys = []
    prefetch = False
    prefetch_execs = 0
    cache_prefetchs = 0
    cache_hits = 0
    cache_hits_graph = ''
    miss = 0
    reads = 0
    pending_requests = 0
    writes = 0
    cache_prefetchs_fails = 0
    syncs = 0
    cachewrite = 0
    curr_exec_asyncs = 0
    pending_requests_fails_time = 0.000000000
    syncs_time = 0.000000000
    cachewrite_time = 0.000000000
    miss_time = 0.000000000
    cache_hits_time = 0.000000000
    pending_requests_time = 0.000000000
    pending_requests_time_res = 0.000000000

    def __init__(self, ksp, table, is_persistent, primary_keys, columns, storage_id=None, storage_class='hecuba.storageobj.StorageObj'):
        """
        Args:
            ksp (str): keyspace name
            table (str): table name
            primary_keys (list(tuple)): a list of (key,type) primary keys (primary + clustering).
            primary_keys (list(tuple)): a list of (key,type) columns
            storage_class (str): full path class name of the storage obj

        Returns:
            None

        """
        dict.__init__(self, {})
        self._storage_id = storage_id
        self.dictCache = PersistentDictCache()
        self._ksp = ksp
        self._table = table
        self.is_persistent = is_persistent
        self._primary_keys = primary_keys
        self._columns = columns
        self._insert_data = None
        self._select_query = {}
        self._batch = None
        self._batchCount = 0
        self._storage_class = storage_class
        if 'type' in self._columns and self._columns['type'] == 'dict':
            self.is_counter = False
        else:
            self.is_counter = self._columns[0][1] == 'counter'

    def init_prefetch(self, block):
        """
        Initializes the prefetch manager
        Args:
           block (hecuba.iter.Block): the dataset partition which need to be prefetch
        Returns:
            None
        """
        self.prefetch = True
        self.prefetchManager = PrefetchManager(1, 1, block)

    def end_prefetch(self):
        self.prefetchManager.terminate()

    def __contains__(self, key):
        if not self.is_persistent:
            return dict.__contains__(self, key)
        else:
            try:
                self.__getitem__(key)
                return True
            except Exception as e:
                print "Exception in persistentDict.__contains__:", e
                return False

    def _preparequery(self, query):
        """
        Internal call for preparing the query for batch insertions

        Args:
          query (str): the query to prepare

        Returns:
            cassandra.query.PreparedStatement: the prepared statement.

        """
        if query in self.prepQueriesDict:
            prepquery = self.prepQueriesDict[query]
            return prepquery
        else:
            try:
                self.prepQueriesDict[query] = config.session.prepare(query)
                prepquery = self.prepQueriesDict[query]
                return prepquery
            except Exception as e:
                print "Error. Couldn't prepare query, retrying: ", e
                self.prepQueriesDict[query] = config.session.prepare(query)
                prepquery = self.prepQueriesDict[query]
                return prepquery

    def __iadd__(self, key, other):
        """
        Implements the += logic.
        This method is consistent only if called on a counter.

        Args:
             key : the key to update
             other: value to add
        Returns:
            None
        """

        if self.is_counter:
            self[key] = other
        else:
            self[key] = self[key] + other

    def __setitem__(self, key, value):
        """
        It handles the writes in the dictionary.
        If the object is persistent, it stores the request in memory up to the point
        they are enough for sending a batch request to Cassandra.
        In case it is not persistent, everything is always held in memory.

        Args:
             key : the key to update
             value: value to set
        Returns:
            None
        """
        self.writes += 1  # STATISTICS
        start = time.time()
        if not self.is_persistent:
            try:
                dict.__setitem__(self, key, value)
            except Exception as e:
                return "Object " + str(self._table) + " with key " + str(key) + " and value " + str(
                    value) + " cannot be inserted in dict" + str(e)
        else:
            if config.cache_activated:
                if len(self.dictCache.cache) >= config.max_cache_size:
                    self._flush_items()
                    self.dictCache.cache = {}
                if key in self.dictCache.cache:
                    val = self.dictCache.cache[key]
                    if val[0] is not None:
                        self.cache_hits += 1
                        # Sync or Sent
                        if val[1] == 'Sync':
                            self.dictCache.sents += 1

                        if self.is_counter:
                            self.dictCache[key] = [int(value) + int(val[0]), 'Sent']
                        else:
                            self.dictCache[key] = [value, 'Sent']
                    else:
                        # Requested
                        self.dictCache[key] = [value, 'Sent']
                        self.dictCache.sents += 1
                else:
                    self.dictCache[key] = [value, 'Sent']
                    self.dictCache.sents += 1
                if self.dictCache.sents == config.batch_size:
                    self.syncs += config.batch_size  # STATISTICS
                    print "self.dictCache:", self.dictCache
                    self._flush_items()
                    end = time.time()  # STATISTICS
                    self.syncs_time += (end - start)  # STATISTICS
                else:
                    self.cachewrite += 1  # STATISTICS
                    end = time.time()  # STATISTICS
                    self.cachewrite_time += (end - start)  # STATISTICS

            else:
                if config.batch_size > 1:
                    if self._batch is None:
                        self._batchCount = 0
                        if self.is_counter:
                            self._batch = BatchStatement(batch_type=BatchType.COUNTER)
                            self._insert_data = self._preparequery(self._build_insert_counter_query())
                        else:
                            self._batch = BatchStatement(batch_type=BatchType.UNLOGGED)
                            self._insert_data = self._preparequery(self._build_insert_query())

                    self._batch.add(self._bind_row(key, value))

                    self._batchCount += 1
                    if self._batchCount % config.batch_size == 0:
                        self._exec_query(self._batch)
                        self._batch = None
                    else:
                        self.cachewrite += 1  # STATISTICS
                        end = time.time()  # STATISTICS
                        self.cachewrite_time += (end - start)  # STATISTICS

                else:
                    if self._insert_data is None:
                        if self.is_counter:
                            self._insert_data = self._preparequery(self._build_insert_counter_query())
                        else:
                            self._insert_data = self._preparequery(self._build_insert_query())

                    query = self._bind_row(key, value)
                    self._exec_query(query)

    def _bind_row(self, key, value):
        """
        Returns:
             BoundStatement: returns a query with binded elements
        """
        elements = []
        if self.is_counter:

            elements.append(int(value))

            if isinstance(key, list) or isinstance(key, tuple):
                for val in key:
                    elements.append(val)
            else:
                elements.append(key)
        else:
            if isinstance(key, list) or isinstance(key, tuple):
                for val in key:
                    elements.append(val)
            else:
                elements.append(key)

            if isinstance(value, list) or isinstance(value, tuple):
                for val in value:
                    elements.append(val)
            else:
                elements.append(value)
        return self._insert_data.bind(elements)

    def _exec_query(self, query):
        """
        This function executes an already binded query
        Args:
            query: binded query already with all data
        Returns:
            None
        """
        start = time.time()
        retry = 5
        while retry > 0:
            try:
                config.session.execute(query)
                retry = 0
            except Exception as e:
                retry -= 1
                logging.warn('retrying  %s', 5 - retry)
                if retry == 0:
                    raise e

        self.syncs += 1  # STATISTICS
        end = time.time()  # STATISTICS
        self.syncs_time += (end - start)  # STATISTICS

    def _flush_items(self):
        """
        This command force the dictionary to write into Cassandra.
        This means that:
          1. if there is insertion stored in the batch, they are executed
          2. anything in the cache is sent to Cassandra
        Returns:
            None
        """
        if self._batch is not None:
            """
            If there are some pending inserts, we flush them to the db. 
            """
            self._exec_query(self._batch)
            self._batch = None

        if len(self.dictCache.cache) == 0:
            return
        if self.is_counter:
            batch = BatchStatement(batch_type=BatchType.COUNTER)
            if self._insert_data is None:
                self._insert_data = self._preparequery(self._build_insert_counter_query())

            for k, v in self.dictCache.cache.iteritems():
                if v[1] == 'Sent':
                    value = v[0]
                    batch.add(self._bind_row(k, int(value)))
                    self.dictCache[k] = [value, 'Sync']
        else:
            if self._insert_data is None:
                self._insert_data = self._preparequery(self._build_insert_query())

            batch = BatchStatement(batch_type=BatchType.UNLOGGED)

            for k, v in self.dictCache.cache.iteritems():
                if v[1] == 'Sent':
                    value = v[0]
                    batch.add(self._bind_row(k, value))
                    self.dictCache[k] = [value, 'Sync']
        self.dictCache.sents = 0

        self._exec_query(batch)

    def __getitem__(self, key):
        """
        If the object is persistent, each request goes to the prefetcher.
        Args:
             key: the dictionary key
        Returns
             item: value found in position key
        """
        logging.debug('GET ITEM %s', key)

        if not self.is_persistent:
            try:
                return dict.__getitem__(self, key)
            except Exception as e:
                return "Object " + str(self.dict_name) + " with key " + str(key) + " not in dict:" + str(e)
        else:
            self.reads += 1
            if config.cache_activated:
                if len(self.dictCache.cache) >= config.max_cache_size:
                    # TODO bad for performance, we should rotate around the cache.
                    self._flush_items()
                    self.dictCache.cache = {}
                if key in self.dictCache.cache:
                    val = self.dictCache.cache[key]
                    item = val[0]
                    return item
                else:
                    if hasattr(self.prefetchManager, 'piper_read'):
                        usedpipe = self.prefetchManager.piper_read[0]
                        askcassandra = True
                        if usedpipe.poll():
                            try:
                                results = usedpipe.recv()
                            except:
                                raise KeyError
                            for entry in results:
                                self.dictCache[entry[0]] = [entry[1], 'Sync']
                            try:
                                item = self.dictCache[key]
                                self.pending_requests += 1
                                return item
                            except KeyError:
                                print "Error obtaining value from Cache"
                                askcassandra = True
                    else:
                        askcassandra = True
                    if askcassandra:
                        self.miss += 1
                        try:
                            item = self._readitem(key)
                        except Exception as e:
                            print "Error:", e
                            raise KeyError
                        self.dictCache[key] = [item, 'Sync']
                        our_dict = self._persistent_props.itervalues().next()
                        other_values = our_dict['columns']
                        if 'type' in other_values:
                            if other_values['type'] == 'dict':
                                items_dict = {}
                                key_names = map(lambda tupla: tupla[0], other_values['primary_keys'])
                                col_names = map(lambda tupla: tupla[0], other_values['columns'])
                                for entry in item:
                                    key_tuple = []
                                    for val in key_names:
                                        found = getattr(entry, val)
                                        key_tuple.append(found)
                                    if len(key_tuple) == 1:
                                        key_tuple = key_tuple[0]
                                    else:
                                        key_tuple = tuple(key_tuple)
                                    val_tuple = []
                                    for val in col_names:
                                        found = getattr(entry, val)
                                        val_tuple.append(found)
                                    if len(val_tuple) == 1:
                                        val_tuple = val_tuple[0]
                                    else:
                                        val_tuple = tuple(val_tuple)
                                    items_dict[key_tuple] = val_tuple
                            else:
                                items_dict = []
                                col_names = map(lambda tupla: tupla[0], other_values)
                                val_tuple = []
                                for entry in item:
                                    for val in col_names:
                                        found = getattr(entry, val)
                                        val_tuple.append(found)
                                    if len(val_tuple) == 1:
                                        val_tuple = val_tuple[0]
                                    else:
                                        val_tuple = tuple(val_tuple)
                                    items_dict.append(val_tuple)
                        else:
                            items_dict = []
                            col_names = map(lambda tupla: tupla[0], other_values)
                            for entry in item:
                                val_tuple = []
                                for val in col_names:
                                    found = getattr(entry, val)
                                    val_tuple.append(found)
                                if len(val_tuple) == 1:
                                    val_tuple = val_tuple[0]
                                else:
                                    val_tuple = tuple(val_tuple)
                                items_dict.append(val_tuple)
                        return items_dict
                        # return item
            else:
                item = self._readitem(key)
                return item

    def _readitem(self, key):
        """
        This function executes de Select query to find the value in position key
        Args:
            key: The position where we want to find the value
        Returns:
            Value: .
        """
        if isinstance(key, tuple):
            quid = len(key)
        else:
            quid = 0
            key = [key]
        if quid not in self._select_query:
            self._select_query[quid] = self._preparequery(self._build_select_query(key))

        query = self._select_query[quid].bind(key)
        result = config.session.execute(query)
        # two conditions different for version 2.x or 3.x of the cassandra drivers
        if (hasattr(result, 'current_rows') and len(result.current_rows) > 1) \
                or (isinstance(result, list) and len(result) > 1):
            item = [row for row in result]
        else:
            item = result[0]
        if type(result) is list:
            if len(result) == 0:
                item = 0
        return item

    def keys(self):
        """
        This method return a list of all the keys of the PersistentDict.
        Returns:
          list: a list of keys
        """
        if not self.is_persistent:
            return dict.keys(self)
        else:
            return self

    def __iter__(self):

        return KeyIter(self._ksp, self._table, self._storage_class, self._storage_id, self._primary_keys)

    def _build_insert_query(self):
        """
        This function builds the insert query
        Args:
            self: dictionary
        Returns:
            str: query string
        """
        query = "INSERT INTO " + self._ksp + "." + self._table + "("
        pk_names = map(lambda tupla: tupla[0], self._primary_keys)
        if 'type' in self._columns and self._columns['type'] == 'dict':
            col_names = map(lambda tupla: tupla[0], self._columns['primary_keys']) +\
                        map(lambda tupla: tupla[0], self._columns['columns'])
            toadd = pk_names + col_names
            query += str.join(',', toadd)
            query += ") VALUES ("
            query += str.join(',', ['?' for i in toadd])
            query += ")"
        else:
            col_names = map(lambda tupla: tupla[0], self._columns)
            toadd = pk_names + col_names
            query += str.join(',', toadd)
            query += ") VALUES ("
            query += str.join(',', ['?' for i in toadd])
            query += ")"
        print "query:", query
        return query

    def _build_select_query(self, key):
        """
        This function builds the insert query
        Args:
            key: list key
        Returns:
            str: query string
        """
        pk_names = map(lambda tupla: tupla[0], self._primary_keys)
        if 'type' in self._columns and self._columns['type'] == 'dict':
            query = "SELECT * FROM " + self._ksp + "." + self._table + " WHERE "
            query += str.join(" AND ", map(lambda k: k + " = ?", pk_names[0:len(key)]))
        else:
            col_names = map(lambda tupla: tupla[0], self._columns)
            selects = str.join(',', pk_names+col_names)
            query = "SELECT " + selects + " FROM " + self._ksp + "." + self._table + " WHERE "
            query += str.join(" AND ", map(lambda k: k + " = ?", pk_names[0:len(key)]))
        return query

    def _build_insert_counter_query(self):
        """
        This function builds the insert query
        Args:
            self: dictionary
        Returns:
            str: query string
        """

        counter_name = self._columns[0][0]
        query = "UPDATE " + self._ksp + "." + self._table + \
                " SET " + counter_name + " = " + counter_name + " + ? WHERE "
        query += str.join(" AND ", map(lambda k: k[0] + " = ?", self._primary_keys))
        return query


class context:
    """
    This Class is used in order to write the last elements found in batch when exiting a worker/loop
    It will access data in a different way depending on if the context is applied on a Block or a PersistentDict:
    If we already have a StorageObj, we'll access it's cache,
    if not, we'll access the cache of the StorageObj of the Block
    """
    def __init__(self, obj):
        """
        Saves the StorageObj in a context attribute
        Args:
            obj: the Block or StorageObj
        """
        if obj.__class__.__name__ == 'Block':
            if obj.storageobj == '':
                print "Error: Block has not a valid storage object"
            else:
                self.storageObj = obj.storageobj
        else:
            self.storageObj = obj

    def __enter__(self):
        """
        Starts the batch in the context
        """
        if self.storageObj.__class__.__name__ == 'PersistentDict':
            self.storageObj.batchvar = True
        else:
            cntxt_dict = self.storageObj._get_default_dict()
            cntxt_dict.batchvar = True

    def __exit__(self):
        """
        Saves the last batch data which hasn't been saved yet and closes the batch
        """
        if self.storageObj.__class__.__name__ == 'PersistentDict':
            midict = self.storageObj
            midict.batchvar = False
            if config.cache_activated:
                micache = midict.dictCache
        else:
            midict = self.storageObj._get_default_dict()
            midict.batchvar = False
            if config.cache_activated:
                micache = midict.dictCache

        if config.cache_activated:
            midict.syncs = midict.syncs + micache.sents
            midict._flush_items()
        else:
            midict.syncs = midict.syncs + midict.batchCount
            midict.config.session.execute(midict.batch)
            midict.batchvar = False