# author: G. Alomar
from cassandra.query import BatchStatement
from cassandra.query import BatchType
from cassandra import ConsistencyLevel
import collections
from hecuba.settings import session, config
from hecuba.cache import PersistentDictCache
from hecuba.prefetchmanager import PrefetchManager
from hecuba.Plist import PersistentKeyList
from conf.hecuba_params import *  # execution_name, batch_size, max_cache_size, cache_activated
from collections import defaultdict
import random
import time
import logging


class PersistentDict(dict):
    """
    This class servers as a proxy to Cassandra. You can access to this class as it was a normal python
    dictionary while under the hood it queries Cassandra.
    """

    prepQueriesDict = {}
    createdColumnfamiliesDict = {}
    types = {}
    insert_data = ''
    batchvar = ''
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

    def __init__(self, mypo, dict_keynames, dict_valnames):
        """
        Args:
            mypo (hecuba.storageobj.StorageObj): the storage object that owns the dictionary
            dict_keynames (list): it is a list of strings containing the names of all keys (primary + clustering).
            doct_valnames (list): a list of the column name of the values.

        Returns:
            None

        """
        dict.__init__(self, {})
        self.dictCache = PersistentDictCache()
        self.mypo = mypo
        self.dict_name = dict_keynames[0]
        self.dict_keynames = dict_keynames
        self.dict_valnames = dict_valnames
        self.insert_data = None
        self.select_query = {}
        self.batch = None
        self.batchCount = 0
        self.keyList = defaultdict(list)

      

    def init_prefetch(self, block):
        """
        Initialized the prefetch manager

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
        if not self.mypo.persistent:
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
                self.prepQueriesDict[query] = session.prepare(query)
                prepquery = self.prepQueriesDict[query]
                return prepquery
            except Exception as e:
                print "query in preparequery:", query
                print "Error. Couldn't prepare query, retrying: ", e
                self.prepQueriesDict[query] = session.prepare(query)
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
        if type(self.dict_name) == tuple:
            counterdictname = self.types[str(self.dict_name[0])]
        else:
            counterdictname = self.types[str(self.dict_name)]
        if counterdictname == 'counter':
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
        if not self.mypo.persistent:
            try:
                dict.__setitem__(self, key, value)
            except Exception as e:
                return "Object " + str(self.dict_name) + " with key " + str(key) + " and value " + str(
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
                        if type(self.dict_name) == tuple:
                            counterdictname = self.types[str(self.dict_name[0])]
                        else:
                            counterdictname = self.types[str(self.dict_name)]
                        if counterdictname == 'counter':
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
                    self._flush_items()
                    end = time.time()  # STATISTICS
                    self.syncs_time += (end - start)  # STATISTICS
                else:
                    self.cachewrite += 1  # STATISTICS
                    end = time.time()  # STATISTICS
                    self.cachewrite_time += (end - start)  # STATISTICS

            else:
                if self.batchvar:
                    if type(self.dict_name) == tuple:
                        counterdictname = self.types[str(self.dict_name[0])]
                    else:
                        counterdictname = self.types[str(self.dict_name)]

                    if self.batch is None:
                        self.batchCount = 0
                        if counterdictname == 'counter':
                            self.batch = BatchStatement(batch_type=BatchType.COUNTER)
                            self.insert_data = self._preparequery(self._build_insert_counter_query())
                        else:
                            self.batch = BatchStatement()
                            self.insert_data = self._preparequery(self._build_insert_query())

                    self.batch.add(self._bind_row(key, value))

                    self.batchCount += 1
                    if self.batchCount % config.batch_size == 0:
                        self._exec_query(self.batch)
                        self.batch = None
                    else:
                        self.cachewrite += 1  # STATISTICS
                        end = time.time()  # STATISTICS
                        self.cachewrite_time += (end - start)  # STATISTICS

                else:
                    if self.insert_data is None:
                        if type(self.dict_name) == tuple:
                            counterdictname = self.types[str(self.dict_name[0])]
                        else:
                            counterdictname = self.types[str(self.dict_name)]
                        if counterdictname == 'counter':
                            self.insert_data = self._preparequery(self._build_insert_counter_query())
                        else:
                            self.insert_data = self._preparequery(self._build_insert_query())

                    query = self._bind_row(key, value)
                    self._exec_query(query)

    def _bind_row(self, key, value):
        """
        
            
        
        Returns:
             BoundStatement: returns a query with binded elements
        """
        elements = []
        if isinstance(key, collections.Iterable):
            for val in key:
                elements.append(val)
        else:
            elements.append(key)

        if isinstance(value, collections.Iterable):
            for val in value:
                elements.append(val)
        else:
            elements.append(key)
        return self.insert_data.bind(elements)

    def _exec_query(self, query):
        done = False
        sessionexecute = 0
        errors = 0
        sleeptime = 0.5
        while not done:
            while sessionexecute < 5:
                try:
                    session.execute(query, d, timeout=10)
                    sessionexecute = 5
                except Exception as e:
                    if sessionexecute == 0:
                        print "sessionexecute Errors in SetItem 2----------------------------"
                    errors += 1
                    time.sleep(sleeptime)
                    if sleeptime < 10:
                        print "sleeptime:", sleeptime, " - Increasing"
                        sleeptime = sleeptime * 1.5 * random.uniform(1.0, 3.0)
                    else:
                        print "sleeptime:", sleeptime, " - Decreasing"
                        sleeptime /= 2
                    if sessionexecute == 4:
                        print "tries:", sessionexecute
                        print "Error: Cannot execute query in setItem. Exception: ", e
                        print "GREPME: The queries were %s" % query
                        print "GREPME: The values were " + str(d)
                        raise e
                    sessionexecute += 1
            done = True
        if errors > 0:
            if errors == 1:
                print "Success after ", errors, " try"
            else:
                print "Success after ", errors, " tries"
        self.syncs += 1  # STATISTICS
        end = time.time()  # STATISTICS
        self.syncs_time += (end - start)  # STATISTICS

    def _flush_items(self):
        """
        This command force the dictionary to write into Cassandra.
        Returns:
            None
        """
        if self.batch is not None:
            """
            If there are some pending inserts, we flush them to the db. 
            """
            self._exec_query(self.batch)
            self.batch = None

        if len(self.dictCache.cache) == 0:
            return
        if type(self.dict_name) == tuple:
            counterdictname = self.types[str(self.dict_name[0])]
        else:
            counterdictname = self.types[str(self.dict_name)]
        if counterdictname == 'counter':
            batch = BatchStatement(batch_type=BatchType.COUNTER)
            if self.insert_data is None:
                self.insert_data = self._preparequery(self._build_insert_counter_query())

            for k, v in self.dictCache.cache.iteritems():
                if v[1] == 'Sent':
                    value = v[0]
                    self.batch.add(self._bind_row(k, value))
                    self.dictCache[k] = [value, 'Sync']
        else:
            if self.insert_data is None:
                self.insert_data = self._preparequery(self._build_insert_query())

            batch = BatchStatement()

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
        """
        logging.debug('GET ITEM %s', key)

        if not self.mypo.persistent:
            try:
                return dict.__getitem__(self, key)
            except Exception as e:
                return "Object " + str(self.dict_name) + " with key " + str(key) + " not in dict:" + str(e)
        else:
            self.reads += 1
            if config.cache_activated:
                if len(self.dictCache.cache) >= config.max_cache_size:
                    # TODO bad for performance, we should rotate aroudn the cache.
                    self._flush_items()
                    self.dictCache.cache = {}
                if key in self.dictCache.cache:
                    val = self.dictCache.cache[key]
                    item = val[0]
                    if len(self.dict_name) == 1:
                        if self.types[str(self.dict_name)] == 'int':
                            item = int(item)
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
                        return item
            else:
                item = self._readitem(key)
                return item

    def _readitem(self, key):
        print "dict.py readitem"
        print "key:", key
        if isinstance(key, tuple):
            quid = len(key)
        else:
            quid=0
        if quid not in self.select_query:
            self.select_query[quid] = self._preparequery(self._build_select_query(key))

        query = self.select_query[quid].bind(key)
        errors = 0
        totalerrors = 0
        sleeptime = 0.5
        done = False
        result = ''
        item = ''
        while not done:
            sessionexecute = 0
            while sessionexecute < 5:
                try:
                    print "       query:", query
                    result = session.execute(query)
                    # two conditions different for version 2.x or 3.x of the cassandra drivers
                    if (hasattr(result, 'current_rows') and len(result.current_rows) > 1) \
                            or (isinstance(result, list) and len(result) > 1):
                        item = [row for row in result]
                    else:
                        item = result[0]
                    sessionexecute = 5
                except Exception as e:
                    print "sleeptime:", sleeptime
                    if sessionexecute == 0:
                        print "       sessionexecute Errors in readitem----------------------------"
                        print "       query:", query
                        errors = 1
                    time.sleep(sleeptime)
                    if sleeptime < 3:
                        sleeptime = sleeptime * 1.5 * random.uniform(1.0, 3.0)
                        totalerrors += 1
                    else:
                        sleeptime /= (totalerrors + 1)
                        sleeptime *= totalerrors
                    if sessionexecute == 4:
                        print "       tries:", sessionexecute
                        print "       GREPME: The queries were %s" % query
                        print "       Error: Cannot execute query in getItem. Exception: ", e
                        raise e
                    sessionexecute += 1
            done = True
        if type(result) is list:
            if len(result) == 0:
                item = 0
                '''
                print "wrong query:", query
                print "type(key):", type(key)
                print "len(result) = 0"
                raise KeyError
                '''
        if errors == 1:
            print "total errors:", totalerrors
        else:
            print "no errors"
        return item


    def keys(self):
        """
        This method return a list of all the keys.

        Returns:
          list: a list of keys
        """
        if not self.mypo.persistent:
            return dict.keys(self)
        else:
            return PersistentKeyList(self)

    def _build_insert_query(self):
        """
        This function builds the insert query
        Args:
            key: dictionary key
            value: value
        Returns:
            str: query string
        """
        query = "INSERT INTO " + self.mypo._ksp + "." + self.mypo._table + "("
        toadd = self.dict_keynames + self.dict_valnames
        query += str.join(',', toadd)
        query += ") VALUES ("
        query += str.join(',', ['?' for i in toadd])
        query += ")"
        return query

    def _build_select_query(self, key):
        """
        This function builds the insert query
        Args:
            key: list key
            value: value
        Returns:
            str: query string
        """

        toadd = self.dict_keynames + self.dict_valnames
        selects = str.join(',', toadd)

        query = "SELECT " + selects +" FROM " + self.mypo._ksp + "." + self.mypo._table+ " WHERE "
        query += str.join(" AND ", map(lambda k: k + " = ?", self.dict_keynames[0:len(key)]))
        return query

    def _build_insert_counter_query(self):
        """
        This function builds the insert query
        Args:
            key: dictionary key
            value: value
        Returns:
            str: query string
        """

        counter_name = self.dict_valnames[0]
        query = "UPDATE " + self.mypo._ksp + "." + self.mypo._table + " SET " + counter_name + " = " + counter_name + " + ? WHERE "
        query += str.join(" AND ", map(lambda k: k + " = ?", self.dict_keynames))
        return query


class context:
    def __init__(self, obj):
        if obj.__class__.__name__ == 'Block':
            if obj.storageobj == '':
                print "Error: Block has not a valid storage object"
            else:
                self.storageObj = obj.storageobj
        else:
            self.storageObj = obj

    def __enter__(self):
        if self.storageObj.__class__.__name__ == 'PersistentDict':
            self.storageObj.batchvar = True
        else:
            keys = self.storageObj.keyList[self.storageObj.__class__.__name__]
            exec ("self.storageObj." + str(keys[0]) + ".batchvar  = True")

    def __exit__(self):
        if self.storageObj.__class__.__name__ == 'PersistentDict':
            midict = self.storageObj
            midict.batchvar = False
            if config.cache_activated:
                micache = midict.dictCache
        else:
            keys = self.storageObj.keyList[self.storageObj.__class__.__name__]
            storobj = self.storageObj
            exec ("midict = storobj." + str(keys[0]))
            midict.batchvar = False
            if config.cache_activated:
                micache = midict.dictCache

        if config.cache_activated:
            midict.syncs = midict.syncs + micache.sents
            midict._flush_items()
        else:
            midict.syncs = midict.syncs + midict.batchCount
            midict.session.execute(midict.batch)
            midict.batchvar = False
