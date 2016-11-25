# author: G. Alomar
from cassandra.query import BatchStatement
from cassandra.query import BatchType
from cassandra import ConsistencyLevel
from hecuba.settings import *
from hecuba.cache import PersistentDictCache
from hecuba.prefetchmanager import PrefetchManager
from hecuba.Plist import PersistentKeyList
from conf.hecuba_params import * # execution_name, batch_size, max_cache_size, cache_activated
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
    keyList = defaultdict(list)
    keyspace = ''
    types = {}
    session = ''
    batch = ""
    batchCount = 0
    firstCounterBatch = 1
    firstNonCounterBatch = 1
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

    def __init__(self, mypo, dict_keynames):
        """
        Args:
            mypo (hecuba.storageobj.StorageObj): the storage object that owns the dictionary
            dict_keynames (list): it is a list of strings containing the names of all keys (primary + clustering).

        Returns:
            None

        """
        dict.__init__(self, {})
        self.dictCache = PersistentDictCache()
        self.mypo = mypo
        self.dict_name = dict_keynames[0]
        if type(dict_keynames) is not list:
            raise TypeError("dict_keynames should be a list")
        self.dict_keynames = dict_keynames

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
            return dict.__contains__(key)
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
          query (string): the query to prepare

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

        self.writes += 1                                               #STATISTICS
        start = time.time()
        if not self.mypo.persistent:
            try:
                dict.__setitem__(self, key, value)
            except Exception as e:
                return "Object " + str(self.dict_name) + " with key " + str(key) + " and value " + str(value) + " cannot be inserted in dict" + str(e)
        else:
            if not 'cache_activated' in globals():
                global cache_activated
                cache_activated = True
            if cache_activated:
                if not 'max_cache_size' in globals():
                    global max_cache_size
                    max_cache_size = 100
                if len(self.dictCache.cache) >= max_cache_size:
                    self._writeitem()
                    self.dictCache.cache = {}
                if key in self.dictCache.cache:
                    val = self.dictCache.cache[key]
                    if val[0] is not None:
                        self.cache_hits += 1
                        #Sync or Sent
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
                        #Requested
                        self.dictCache[key] = [value, 'Sent']
                        self.dictCache.sents += 1
                else:
                    self.dictCache[key] = [value, 'Sent']
                    self.dictCache.sents += 1
                if not 'batch_size' in globals():
                    global batch_size
                    batch_size = 100
                if self.dictCache.sents == batch_size:
                    self.syncs = self.syncs + batch_size                            #STATISTICS
                    self._writeitem()
                    end = time.time()                                               #STATISTICS
                    self.syncs_time += (end-start)                                  #STATISTICS
                else:
                    self.cachewrite += 1                                            #STATISTICS
                    end = time.time()                                               #STATISTICS
                    self.cachewrite_time += (end-start)                             #STATISTICS

            else:
                if self.batchvar:
                    if type(self.dict_name) == tuple:
                        counterdictname = self.types[str(self.dict_name[0])]
                    else:
                        counterdictname = self.types[str(self.dict_name)]
                    if counterdictname == 'counter':
                        query = "UPDATE " + self.mypo._ksp + ".\"" + self.mypo._table + "\" SET " + self.dict_name + " = " + self.dict_name + " + ? WHERE "
                        if self.firstCounterBatch == 1:
                            self.batchCount = 0
                            self.batch = BatchStatement(batch_type=BatchType.COUNTER)
                            self.firstCounterBatch = 0
                            if not type(self.dict_keynames) is tuple:
                                query += self.dict_keynames + " = ?"
                            else:
                                for i, k in enumerate(self.dict_keynames):
                                    if i < (len(self.dict_keynames) - 1):
                                        query += str(self.dict_keynames[i]) + " = ? AND "
                                    else:
                                        query += str(self.dict_keynames[i]) + " = ?"
                            self.insert_data = self._preparequery(query)

                        query = "self.batch.add(self.insert_data, ("
                        query += str(value) + ", "
                        if type(key) == tuple:
                            for ind, val in enumerate(key):
                                query += "" + str(key[ind]) + ", "
                        else:
                            if self.types[str(self.dict_keynames)] == 'text':
                                query += "\'" + str(key) + "\'))"
                            else:
                                query += str(key) + "))"

                    else:
                        query = "INSERT INTO " + self.mypo._ksp + ".\"" + self.mypo._table + "\"("
                        stringkeynames = str(self.dict_keynames)
                        stringkeynames = stringkeynames.replace('\'', '')
                        stringkeynames = stringkeynames.replace('(', '')
                        stringkeynames = stringkeynames.replace(')', '')
                        if self.firstNonCounterBatch == 1:
                            self.batchCount = 0
                            self.firstNonCounterBatch = 0
                            self.batch = BatchStatement(consistency_level=ConsistencyLevel.QUORUM)

                            if not type(self.dict_keynames) is tuple:
                                query += stringkeynames + ", "
                            else:
                                for k in self.dict_keynames:
                                    query += str(k) + ","

                            if not type(self.dict_name) is tuple:
                                query += self.dict_name
                            else:
                                for ind, val in self.dict_name:
                                    if ind < (len(self.dict_name) - 1):
                                        query += str(val) + ", "
                                    else:
                                        query += str(val)

                            query += ") VALUES ("

                            if not type(key) is tuple:
                                query += "?, "
                            else:
                                for i in range(0, len(key)):
                                    query += "?, "
                            if not type(value) is tuple:
                                query += "?)"
                            else:
                                for ind in range(0, len(value)):
                                    if ind < (len(value) - 1):
                                        query += "?, "
                                    else:
                                        query += "?)"

                            self.insert_data = self._preparequery(query)

                        query = "self.batch.add(self.insert_data, ("
                        if type(key) == tuple:
                            for ind, val in enumerate(key):
                                if ind < (len(key) - 1):
                                    if self.types[str(self.dict_keynames[ind])] == 'text':
                                        query += "\'" + str(key[ind]) + "\', "
                                    else:
                                        query += str(key[ind]) + ", "
                                else:
                                    if self.types[str(self.dict_keynames[ind])] == 'text':
                                        query += "\'" + str(key[ind]) + "\'"
                                    else:
                                        query += str(key[ind])
                        else:
                            if self.types[str(self.dict_keynames)] == 'text':
                                query += "\'" + str(key) + "\'"
                            else:
                                query += str(key)
                        if type(value) == tuple:
                            for ind, val in enumerate(value):
                                if ind < (len(key) - 1):
                                    if self.types[str(self.dict_name)] == 'text':
                                        query += ", \'" + str(value[ind]) + "\'"
                                    else:
                                        query += ", " + int(value[ind])
                                else:
                                    if self.types[str(self.dict_name)] == 'text':
                                        query += ", \'" + str(value[ind]) + "\'"
                                    else:
                                        query += ", " + int(value[ind]) + "))"
                        else:
                            if self.types[str(self.dict_name)] == 'text':
                                query += ", \'" + str(value) + "\'))"
                            else:
                                query += ", " + str(value) + "))"
                    exec query

                    self.batchCount += 1

                    done = False
                    sessionexecute = 0
                    if not 'batch_size' in globals():
                        global batch_size
                        batch_size = 100
                    if self.batchCount % batch_size == 0:
                        errors = 0
                        sleeptime = 0.5
                        while not done:
                            while sessionexecute < 10:
                                try:
                                    session.execute(self.batch)
                                    self.firstCounterBatch = 1
                                    self.firstNonCounterBatch = 1
                                    sessionexecute = 10
                                except Exception as e:
                                    if sessionexecute == 0:
                                        print "sessionexecute Errors in SetItem 1----------------------------"
                                    errors += 1
                                    time.sleep(sleeptime)
                                    if sleeptime < 40:
                                        print "sleeptime:", sleeptime, " - Increasing"
                                        sleeptime = sleeptime * 1.5 * random.uniform(1.0, 3.0)
                                    else:
                                        print "sleeptime:", sleeptime, " - Decreasing"
                                        sleeptime /= 2
                                    if sessionexecute == 9:
                                        print "tries:", sessionexecute
                                        print "Error: Cannot execute query in setItem. Exception: ", e
                                        print "GREPME: The queries were %s"%(query)
                                        print "GREPME: The values were " + str(d)
                                        raise e
                                    sessionexecute += 1
                            done = True
                        if errors > 0:
                            if errors == 1:
                                print "Success after ", errors, " try"
                            else:
                                print "Success after ", errors, " tries"
                        self.syncs += batch_size                                    #STATISTICS
                        end = time.time()                                           #STATISTICS
                        self.syncs_time += (end-start)                              #STATISTICS
                    else:
                        self.cachewrite += 1                                        #STATISTICS
                        end = time.time()                                           #STATISTICS
                        self.cachewrite_time += (end-start)                         #STATISTICS

                else:
                    d = {}
                    if type(self.dict_name) == tuple:
                        counterdictname = self.types[str(self.dict_name[0])]
                    else:
                        counterdictname = self.types[str(self.dict_name)]
                    if counterdictname == 'counter':
                        query = "UPDATE " + self.keyspace + ".\"" + self.mypo.name + "\" SET "
                        query += "\"" + self.dict_name + "\" = \"" + self.dict_name + "\" + %(val)s WHERE "
                        if not type(self.dict_keynames) is tuple:
                            query += self.dict_keynames + " = %(key)s"
                            if self.types[str(self.dict_keynames)] == 'text':
                                d["key"] = str(key)
                            else:
                                d["key"] = int(key)
                        else:
                            for i, k in enumerate(self.dict_keynames):
                                ikey = i + 1
                                if ikey < (len(self.dict_keynames) - 1):
                                    query += str(self.dict_keynames[i]) + " = %(key" + str(ikey) + ")s AND "
                                else:
                                    query += str(self.dict_keynames[i]) + " = %(key" + str(ikey) + ")s"
                                if type(key) is int:
                                    d["key" + str(ikey)] = int(key)
                                else:
                                    d["key" + str(ikey)] = str(key)
                        if self.types[str(self.dict_name)] == 'text':
                            d["val"] = "" + str(value) + ""
                        else:
                            d["val"] = int(value)
                    else:
                        query = "INSERT INTO " + self.keyspace + ".\"" + self.mypo.name + "\"("
                        if not type(key) is tuple:
                            stringkeynames = str(self.dict_keynames)
                            stringkeynames = stringkeynames.replace('\'', '')
                            stringkeynames = stringkeynames.replace('(', '')
                            stringkeynames = stringkeynames.replace(')', '')
                            query += stringkeynames + ", "
                        else:
                            for i, k in enumerate(key):
                                query += self.dict_keynames[i] + ", "
                        if type(self.dict_name) == tuple:
                            for ival, val in enumerate(self.dict_name):
                                if ival < (len(self.dict_name) - 1):
                                    query += str(val) + ", "
                                else:
                                    query += str(val)
                        else:
                            query += str(self.dict_name)
                        query += ") VALUES ("
                        d = {}
                        if not type(key) is tuple:
                            if self.types[str(self.dict_keynames)] == 'text':
                                d["k"] = str(key)
                            else:
                                d["k"] = int(key)
                            query += "%(k)s, "
                        else:
                            for i, k in enumerate(key):
                                query += "%(k" + str(i) + ")s, "
                                if self.types[str(self.dict_keynames[i])] == 'text':
                                    d["k" + str(i)] = str(k)
                                else:
                                    d["k" + str(i)] = int(k)
                        if not type(value) is tuple:
                            query += "%(val)s)"
                            if self.types[str(self.dict_name)] == 'text':
                                d["val"] = str(value)
                            else:
                                d["val"] = int(value)
                        else:
                            for i, v in enumerate(value):
                                if self.types[str(self.dict_name[i])] == 'text':
                                    d["val" + str(i)] = str(v)
                                else:
                                    d["val" + str(i)] = int(v)
                                if i < (len(self.dict_name) - 1):
                                    query += "%(val" + str(i) + ")s, "
                                else:
                                    query += "%(val" + str(i) + ")s)"

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
                    self.syncs += 1                                                 #STATISTICS
                    end = time.time()                                               #STATISTICS
                    self.syncs_time += (end-start)                                  #STATISTICS

    def _writeitem(self):
        """
        This command force the dictionary to write into Cassandra.
        Returns:
            None
        """
        if len(self.dictCache.cache) == 0:
            #print "Nothing to write, cache empty"
            return
        if type(self.dict_name) == tuple:
            counterdictname = self.types[str(self.dict_name[0])]
        else:
            counterdictname = self.types[str(self.dict_name)]
        if counterdictname == 'counter':
            query = "UPDATE " + self.mypo._ksp + ".\"" + self.mypo._table + "\" SET " + self.dict_name + " = " + self.dict_name + " + ? WHERE "
            self.batch = BatchStatement(batch_type=BatchType.COUNTER)
            if not type(self.dict_keynames) is tuple:
                query += self.dict_keynames + " = ?"
            else:
                for i, k in enumerate(self.dict_keynames):
                    if i < (len(self.dict_keynames) - 1):
                        query += str(self.dict_keynames[i]) + " = ? AND "
                    else:
                        query += str(self.dict_keynames[i]) + " = ?"


            self.insert_data = self._preparequery(query)

            query1 = "self.batch.add(self.insert_data, ("
            for k, v in self.dictCache.cache.iteritems():
                if v[1] == 'Sent':
                    value = v[0]
                    if type(value) == tuple:
                        for ind, val in enumerate(value):
                            if self.types[str(self.dict_name[ind])] == 'text':
                                query = query1 + "\'" + str(val) + "\',"
                            else:
                                query = query1 + str(val) + ","
                    else:
                        if self.types[str(self.dict_name)] == 'text':
                            query = query1 + "\'" + str(value) + "\',"
                        else:
                            query = query1 + str(value) + ","
                    if type(k) == tuple:
                        for ind, val in enumerate(k):
                            if ind < (len(k) - 1):
                                if self.types[str(self.dict_keynames[ind])] == 'text':
                                    query += "\'" + str(k[ind]) + "\', "
                                else:
                                    query += str(k[ind]) + ", "
                            else:
                                if self.types[str(self.dict_keynames[ind])] == 'text':
                                    query += "\'" + str(k[ind]) + "\'))"
                                else:
                                    query += str(k[ind]) + "))"
                    else:
                        if self.types[str(self.dict_keynames)] == 'text':
                            query += "\'" + str(k) + "\'))"
                        else:
                            query += str(k) + "))"
                    exec query
                    self.dictCache[k] = [value, 'Sync']
        else:
            query = "INSERT INTO " + self.mypo._ksp + ".\"" + self.mypo._table + "\"("
            stringkeynames = str(self.dict_keynames)
            stringkeynames = stringkeynames.replace('\'', '')
            stringkeynames = stringkeynames.replace('(', '')
            stringkeynames = stringkeynames.replace(')', '')
            self.batch = BatchStatement(consistency_level=ConsistencyLevel.QUORUM)

            if not type(self.dict_keynames) is tuple:
                query += stringkeynames + ", "
            else:
                for i, k in enumerate(self.dict_keynames):
                    query += self.dict_keynames[i] + ","

            if not type(self.dict_name) is tuple:
                query += self.dict_name
            else:
                for i, v in enumerate(self.dict_name):
                    if i < (len(self.dict_name) - 1):
                        query += self.dict_name[i] + ","
                    else:
                        query += self.dict_name[i]

            query += ") VALUES ("

            if not type(self.dict_keynames) is tuple:
                query += "?, "
            else:
                for i in range(0, len(self.dict_keynames)):
                    query += "?, "

            if not type(self.dict_name) is tuple:
                query += "?)"
            else:
                for ind, v in enumerate(self.dict_name):
                    if ind < (len(self.dict_name) - 1):
                        query += "?, "
                    else:
                        query += "?)"

            query += ";"
            self.insert_data = self._preparequery(query)

            query1 = "self.batch.add(self.insert_data, ("
            for k, v in self.dictCache.cache.iteritems():
                if v[1] == 'Sent':
                    value = v[0]
                    if type(self.dict_keynames) == tuple:
                        query = query1
                        for ind, val in enumerate(k):
                            if ind < (len(k) - 1):
                                if self.types[str(self.dict_keynames[ind])] == 'text':
                                    query += "\'" + str(k[ind]) + "\', "
                                else:
                                    query += str(k[ind]) + ", "
                            else:
                                if self.types[str(self.dict_keynames[ind])] == 'text':
                                    query += "\'" + str(k[ind]) + "\'"
                                else:
                                    query += str(k[ind])
                    else:
                        if self.types[str(self.dict_keynames)] == 'text':
                            query = query1 + "\'" + str(k) + "\'"
                        else:
                            query = query1 + str(k) + ""

                    if type(value) == list:
                        newv = value
                        value = newv[0]
                    if type(self.dict_name) == tuple:
                        for ind, val in enumerate(value):
                            if ind < (len(value) - 1):
                                if self.types[str(self.dict_name[ind])] == 'text':
                                    query += ", \'" + str(value[ind]) + "\'"
                                else:
                                    query += ", " + str(value[ind])
                            else:
                                if self.types[str(self.dict_name[ind])] == 'text':
                                    query += ", \'" + str(value[ind]) + "\'))"
                                else:
                                    query += ", " + str(value[ind]) + "))"
                    else:
                        if self.types[str(self.dict_name)] == 'text':
                            query += ", \'" + str(value) + "\'))"
                        else:
                            query += ", " + str(value) + "))"
                    exec query
                    self.dictCache[k] = [value, 'Sync']
        self.dictCache.sents = 0

        done = False
        sessionexecute = 0
        errors = 0
        sleeptime = 0.5
        while not done:
            while sessionexecute < 5:
                try:
                    session.execute(self.batch)
                    self.firstCounterBatch = 1
                    self.firstNonCounterBatch = 1
                    sessionexecute = 5
                except Exception as e:
                    if sessionexecute == 0:
                        print "sessionexecute Errors in SetItem 3----------------------------"
                    errors += 1
                    time.sleep(sleeptime)
                    if sleeptime < 40:
                        print "sleeptime:", sleeptime, " - Increasing"
                        sleeptime = sleeptime * 1.5 * random.uniform(1.0, 3.0)
                    else:
                        print "sleeptime:", sleeptime, " - Decreasing"
                        sleeptime /= 2
                    if sessionexecute == 4:
                        print "tries:", sessionexecute
                        print "Error: Cannot execute query in setItem. Exception: ", e
                        print "GREPME: The queries were %s" % query
                        raise e
                    sessionexecute += 1
            done = True
        if errors > 0:
            if errors == 1:
                print "Success after ", errors, " try"
            else:
                print "Success after ", errors, " tries"

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
            if not 'cache_activated' in globals():
                global cache_activated
                cache_activated = True
            if cache_activated:
                if not 'max_cache_size' in globals():
                    global max_cache_size
                    max_cache_size = 100
                if len(self.dictCache.cache) >= max_cache_size:
                    self._writeitem()
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
        query = "SELECT "
        columns = list(self.dict_keynames) + list(self.dict_name)
        if len(columns) > 1:
            for ind, val in enumerate(columns):
                if ind < (len(columns) - 1):
                    query += str(columns[ind]) + ", "
                else:
                    query += str(columns[ind])
        else:
            query += str(columns[0])
        query += " FROM " + self.mypo._ksp + ".\"" + self.mypo._table + "\" WHERE "
        if not type(key) is tuple:
            key = str(key).replace("[", "")
            key = str(key).replace("]", "")
            key = str(key).replace("'", "")

            if isinstance(self.dict_keynames,tuple):
                keyname = self.dict_keynames[0]
            else:
                keyname = self.dict_keynames

            if self.types[str(keyname)] == 'text':
                query += keyname + " = \'" + str(key)
            else:
                query += keyname + " = " + str(key)
        else:
            for i, k in enumerate(key):
                if i < (len(key) - 1):
                    if self.types[str(self.dict_keynames[i])] == 'text':
                        query += self.dict_keynames[i] + " = \'" + str(k) + "\' AND "
                    else:
                        query += self.dict_keynames[i] + " = " + str(k) + " AND "
                else:
                    if self.types[str(self.dict_keynames[i])] == 'text':
                        query += self.dict_keynames[i] + " = \'" + str(k)
                    else:
                        query += self.dict_keynames[i] + " = " + str(k)
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
                    #two conditions different for version 2.x or 3.x of the cassandra drivers
                    if (hasattr(result,'current_rows') and len(result.current_rows) > 1) \
                            or (isinstance(result,list) and len(result) > 1):
                        item = [row for row in result]
                    else:
                        item = ''
                        for row in result:
                            print "row:", row
                            if len(row) > 1:
                                item = []
                                for i, val in enumerate(row):
                                    print "i:  ", i
                                    print "val:", val
                                    item.append(val)
                            else:
                                item = val
                    print "item:", str(item)
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
        '''
        if (self.types[str(self.dict_name)] == 'int') or (self.types[str(self.dict_name)] == 'counter') or (len(result) == 0):
            print "if"
            return int(item)
        else:
            print "else"
            return item
        '''

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
            exec("self.storageObj." + str(keys[0]) + ".batchvar  = True")

    def __exit__(self):
        if self.storageObj.__class__.__name__ == 'PersistentDict':
            midict = self.storageObj
            midict.batchvar = False
            if not 'cache_activated' in globals():
                global cache_activated
                cache_activated = True
            if cache_activated:
                micache = midict.dictCache
        else:
            keys = self.storageObj.keyList[self.storageObj.__class__.__name__]
            storobj = self.storageObj
            exec("midict = storobj." + str(keys[0]))
            midict.batchvar = False
            if not 'cache_activated' in globals():
                global cache_activated
                cache_activated = True
            if cache_activated:
                micache = midict.dictCache

        if cache_activated:
            midict.syncs = midict.syncs + micache.sents
            midict._writeitem()
        else:
            midict.syncs = midict.syncs + midict.batchCount
            midict.session.execute(midict.batch)
            midict.batchvar = False
