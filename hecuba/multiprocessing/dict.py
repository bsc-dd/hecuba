from cassandra.query import SimpleStatement
from cassandra.query import BatchStatement
from cassandra.query import BatchType
from cassandra import ConsistencyLevel
from pyapi.cache import PersistentDictCache
from pyapi.prefetchmanager import PrefetchManager
from pyapi.datastore import *
from pyapi.list import KeyList
from pyapi.Plist import PersistentKeyList
from collections import OrderedDict
from collections import defaultdict
import random
import time
import pickle
import sys
import os
import threading
from app.params import execution_name, batch_size, prefetch_size, prefetch_distance, max_cache_size
from pprint import pprint

class PersistentDict(dict):

    prepQueriesDict = {}
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
    dictCache = PersistentDictCache()
    prefetchManager = ""
    blockKeys = []
    prefetch = False
    use_cache = True         #MODIFY HERE
    #statistics parameters
    prefetch_execs              = 0
    cache_prefetchs             = 0
    cache_hits                  = 0
    cache_hits_graph            = ''
    miss                        = 0
    reads                       = 0
    pending_requests            = 0
    writes                      = 0
    cache_prefetchs_fails       = 0
    syncs                       = 0
    cachewrite                  = 0
    curr_exec_asyncs            = 0
    pending_requests_fails_time = 0.000000000
    syncs_time                  = 0.000000000
    cachewrite_time             = 0.000000000
    miss_time                   = 0.000000000
    cache_hits_time             = 0.000000000
    pending_requests_time       = 0.000000000
    pending_requests_time_res   = 0.000000000

    def __init__(self, mypo = None, dict_name = None, dict_keynames = None):
        dict.__init__(self, {})

        self.mypo = mypo
        self.dict_name = dict_name
        self.dict_keynames = dict_keynames
        self.prefetchManager = PrefetchManager(self.mypo.cluster, 4) #max_conc_proc=32

    def init_prefetch(self, block):
        self.prefetch = True
        for key in block:
            self.blockKeys.append(key)

    def __contains__(self,key):
        if self.mypo.persistent == False:
            return dict.__contains__(key)
        else:
            try:
                self.__getitem__(key)
                return True
            except Exception as e:
                return False

    def prepareQuery(self,query):
        try:
            prepQuery = self.prepQueriesDict[query]
            return prepQuery
        except Exception as e:
            print "PREPARED_QUERY:", query
            self.prepQueriesDict[query] = self.session.prepare(query)
            prepQuery = self.prepQueriesDict[query]
            return prepQuery

    def __setitem__(self, key, value):
        self.writes = self.writes + 1                                               #STATISTICS
        start = time.time()
        if self.mypo.persistent == False:
            try:
                dict.__setitem__(self, key, value)
            except Exception as e:
                return "Object " + str(self.dict_name) + " with key " + str(key) + " and value " + str(value) + " cannot be inserted in dict"
        else:
            if self.use_cache == True:
                if len(self.dictCache.cache) >= max_cache_size:
                    #print "cache full!"
                    self.writeitem()
                    self.dictCache.cache = {}
	        if key in self.dictCache.cache:
                    val = self.dictCache.cache[key]
                    if not val[0] == None:
			#Sync or Sent
                        if val[1] == 'Sync':
                            self.dictCache.sents += 1
                        self.dictCache[key] = [value, 'Sent']
                    else:
			#Requested
                        self.dictCache[key] = [value, 'Sent']
                        self.dictCache.sents += 1
                else:
                    self.dictCache[key] = [value, 'Sent']
                    self.dictCache.sents += 1
                if self.dictCache.sents == batch_size:
                    self.syncs = self.syncs + batch_size                            #STATISTICS
                    self.writeitem()
                    end = time.time()                                               #STATISTICS
                    self.syncs_time += (end-start)                                  #STATISTICS
                else:
                    self.cachewrite = self.cachewrite + 1                           #STATISTICS
                    end = time.time()                                               #STATISTICS
                    self.cachewrite_time += (end-start)                             #STATISTICS
                    #pass
                    
            else:
                session = self.session
                if self.batchvar == 'true':
                    if type(self.dict_name) == tuple:
                        counterDictName = self.types[str(self.dict_name[0])]
                    else:
                        counterDictName = self.types[str(self.dict_name)]
		    if counterDictName == 'counter':
	                query = "UPDATE " + self.keyspace + ".\"" + self.mypo.name + "\" SET " + self.dict_name + " = " + self.dict_name + " + ? WHERE "
	                if self.firstCounterBatch == 1:
	                    self.batchCount = 0
	                    self.batch = BatchStatement(batch_type=BatchType.COUNTER)
	                    self.firstCounterBatch = 0
	                    if not type(self.dict_keynames) is tuple:
	                        query+= self.dict_keynames + " = ?"
	                    else:
	                        lenk = len(self.dict_keynames) - 1
	                        for i, k in enumerate(self.dict_keynames):
	                            if i < lenk:
	                                query+= str(self.dict_keynames[i]) + " = %(key" + str(i) + ")s AND "
	                            else:
	                                query+= str(self.dict_keynames[i]) + " = %(key" + str(i) + ")s"
	                    self.insert_data = self.prepareQuery(query)

	                query = "self.batch.add(self.insert_data, ("
	                query += str(value) + ", "
	                if type(key) == tuple:
	                    lenk = len(key) - 1
	                    for ind,val in enumerate(key):
	                        query += "" + str(key[ind]) + ", "
	                else:
		            if self.types[str(self.dict_keynames)] == 'text':
	                        query += "\'" + str(key) + "\'))"
		            else:
	                        query += ""   + str(key) +   "))"

	            else:
                        query = "INSERT INTO " + self.keyspace + ".\"" + self.mypo.name + "\"("
                        stringKeynames = str(self.dict_keynames)
                        stringKeynames = stringKeynames.replace('\'','')
                        stringKeynames = stringKeynames.replace('(','')
                        stringKeynames = stringKeynames.replace(')','')
	                if self.firstNonCounterBatch == 1:
	                    self.batchCount = 0
                            self.firstNonCounterBatch = 0
	                    self.batch = BatchStatement(consistency_level=ConsistencyLevel.QUORUM)

		            if not type(self.dict_keynames) is tuple:
		                query+= stringKeynames + ", "
	                    else:
	 	                for k in self.dict_keynames:
		                    query += str(k) + ","

                            if not type(self.dict_name) is tuple:
                                query += self.dict_name
                            else:
                                lenv = len(self.dict_name) - 1
                                for ind,val in self.dict_name:
                                    if ind < lenv:
                                        query += str(val) + ", "
                                    else:
                                        query += str(val)

                            query += ") VALUES ("

	    	            if not type(key) is tuple:
	                        query += "?, "
	                    else:
	                        for i,k in enumerate(key):
	                            query += "%(k" + str(i) + ")s, "
                            if not type(value) is tuple:
                                query += "?)"
                            else:
                                lenv = len(value) - 1
                                for ind, v in enumerate(value):
                                    if ind < lenv:
                                        query += "?, "
                                    else:
                                        query += "?)"

                            self.insert_data = self.prepareQuery(query)

	                query = "self.batch.add(self.insert_data, ("
	                if type(key) == tuple:
	                    lenk = len(key) - 1
	                    for ind,val in enumerate(key):
	                        if ind < lenk:
	                            query += "\'" + str(key[ind]) + "\', "
	                        else:
	                            query += "\'" + str(key[ind]) + "\'"
	                else:
                            if self.types[str(self.dict_keynames)] == 'text':
			        query += "\'" + str(key) + "\'"
                            else:
			        query += "" + str(key) + ""
                        if type(value) == tuple:
                            lenv = len(value) - 1
                            for ind,val in enumerate(value):
	                        if ind < lenk:
                                    if self.types[str(self.dict_name)] == 'text':
	                                query += ", \'" + str(value[ind]) + "\'"
                                    else:
	                                query += ", " + int(value[ind]) + ""
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
	            exec(query)

	            self.batchCount += 1
                
	            done = False
	            sessionExecute = 0
	            if self.batchCount % batch_size == 0:
                        errors = 0
                        sleepTime = 0.5
	                while done == False:
	                    while sessionExecute < 10:
	                        try:
	                            session.execute(self.batch) 
	                            self.firstCounterBatch = 1
	                            self.firstNonCounterBatch = 1
	                            sessionExecute = 10
	                        except Exception as e:
	                            if sessionExecute == 0:
	                                print "sessionExecute Errors in SetItem----------------------------"
	                            errors = errors + 1
	                            time.sleep(sleepTime)
	                            if sleepTime < 40:
	                                print "sleepTime:", sleepTime, " - Increasing"
	                                sleepTime = sleepTime * 1.5 * random.uniform(1.0, 3.0)
	                            else:
	                                print "sleepTime:", sleepTime, " - Decreasing"
	                                sleepTime = sleepTime / 2
	                            if sessionExecute == 9:
	                                print "tries:", sessionExecute
	                                print "Error: Cannot execute query in setItem. Exception: ", e
	                                print "GREPME: The queries were %s"%(query)
	                                print "GREPME: The values were " + str(d)
	                                raise e
	                            sessionExecute = sessionExecute + 1
	                    done = True
	                if errors > 0:
	                    if errors == 1:
	                        print "Success after ", errors, " try"
	                    else:
	                        print "Success after ", errors, " tries"
	                    errors = 0
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
                        counterDictName = self.types[str(self.dict_name[0])]
                    else:
                        counterDictName = self.types[str(self.dict_name)]
		    if counterDictName == 'counter':
		        query = "UPDATE " + self.keyspace + ".\"" + self.mypo.name + "\" SET "
		        query += "\"" + self.dict_name + "\" = \"" + self.dict_name + "\" + %(val)s WHERE "
		        if not type(self.dict_keynames) is tuple:
		            query += self.dict_keynames + " = %(key)s"
                            if self.types[str(self.dict_keynames)] == 'text':
		                d["key"] = str(key)
		            else:
		                d["key"] = int(key)
		        else:
		            lenk = len(self.dict_keynames) - 1
		            for i,k in enumerate(self.dict_keynames):
		                ikey = i + 1
		                if ikey < lenk:
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
                            stringKeynames = str(self.dict_keynames)
                            stringKeynames = stringKeynames.replace('\'','')
                            stringKeynames = stringKeynames.replace('(','')
                            stringKeynames = stringKeynames.replace(')','')
			    query+= stringKeynames + ", "
		        else:
		 	    for i,k in enumerate(key):
		               query += self.dict_keynames[i] + ", "
                        if type(self.dict_name) == tuple:
		            lenv = len(self.dict_name) - 1
                            for ival,val in enumerate(self.dict_name):
		                if ival < lenv:
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
		            for i,k in enumerate(key):
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
                            for i,v in enumerate(value):
                                if self.types[str(self.dict_name[i])] == 'text':
                                    d["val" + str(i)] = str(v)
                                else:
                                    d["val" + str(i)] = int(v)
                                if i < lenv:
		                    query += "%(val" + str(i) + ")s, "
                                else:
		                    query += "%(val" + str(i) + ")s)"

		    done = False
		    sessionExecute = 0
                    errors = 0
                    sleepTime = 0.5
		    while done == False:
		        while sessionExecute < 5:
		            try:
		                session.execute(query, d, timeout=10)
		                sessionExecute = 5
		            except Exception as e:
		                if sessionExecute == 0:
		                    print "sessionExecute Errors in SetItem----------------------------"
		                errors = errors + 1
		                time.sleep(sleepTime)
		                if sleepTime < 10:
		                    print "sleepTime:", sleepTime, " - Increasing"
		                    sleepTime = sleepTime * 1.5 * random.uniform(1.0, 3.0)
		                else:
		                    print "sleepTime:", sleepTime, " - Decreasing"
		                    sleepTime = sleepTime / 2
		                if sessionExecute == 4:
		                    print "tries:", sessionExecute
		                    print "Error: Cannot execute query in setItem. Exception: ", e
		                    print "GREPME: The queries were %s"%(query)
		                    print "GREPME: The values were " + str(d)
		                    raise e
		                sessionExecute = sessionExecute + 1
		        done = True
		    if errors > 0:
		        if errors == 1:
		            print "Success after ", errors, " try"
		        else:
		            print "Success after ", errors, " tries"
		        errors = 0
                    self.syncs += 1                                                 #STATISTICS
                    end = time.time()                                               #STATISTICS
                    self.syncs_time += (end-start)                                  #STATISTICS

    def writeitem(self):
            session = self.session
            if type(self.dict_name) == tuple:
                counterDictName = self.types[str(self.dict_name[0])]
            else:
                counterDictName = self.types[str(self.dict_name)]
            if counterDictName == 'counter':
		query = "UPDATE " + self.keyspace + ".\"" + self.mypo.name + "\" SET " + self.dict_name + " = " + self.dict_name + " + ? WHERE "
		self.batch = BatchStatement(batch_type=BatchType.COUNTER)
		if not type(self.dict_keynames) is tuple:
		    query+= self.dict_keynames + " = ?"
		else:
		    lenk = len(self.dict_keynames) - 1
		    for i,k in enumerate(self.dict_keynames):
			ikey = i + 1
			if ikey < lenk:
			    query+= str(self.dict_keynames[i]) + " = %(key" + str(ikey) + ")s AND "
			else:
			    query+= str(self.dict_keynames[i]) + " = %(key" + str(ikey) + ")s"

		self.insert_data = self.prepareQuery(query)

	        query1 = "self.batch.add(self.insert_data, ("
	        for k,v in self.dictCache.cache.iteritems():
		    if v[1] == 'Sent':
		        value = v[0]
                        '''
		        if type(value) == list:
			    newv = value
			    value = newv[0]
                        '''
                        if type(value) == tuple:
                            for ind,val in enumerate(value):
                                if self.types[str(self.dict_name[ind])] == 'text':
                                    query = query1 + "\'" + str(val) + "\',"
                                else:
                                    query = query1 + ""   + str(val) +   ","
                        else:
		            if self.types[str(self.dict_name)] == 'text':
			        query = query1 + "\'" + str(value) + "\',"
		            else:
			        query = query1 + ""   + str(value) +   ","
		        if type(k) == tuple:
			    lenk = len(k) - 1
			    for ind,val in enumerate(k):
			        if ind < lenk:
			            if self.types[str(self.dict_keynames[ind])] == 'text':
				        query += "\'" + str(k[ind]) + "\', "
                                    else:
				        query += ""   + str(k[ind]) +   ", "
			        else:
			            if self.types[str(self.dict_keynames[ind])] == 'text':
				        query += "\'" + str(k[ind]) + "\'))"
                                    else:
				        query += ""   + str(k[ind]) +   "))"
		        else:
			    if self.types[str(self.dict_keynames)] == 'text':
			        query += "\'" + str(k) + "\'))"
			    else:
			        query += ""   + str(k) +   "))" 
		        exec(query)
		        self.dictCache[k]=[value,'Sync']
	    else:
		query = "INSERT INTO " + self.keyspace + ".\"" + self.mypo.name + "\"("
		stringKeynames = str(self.dict_keynames)
		stringKeynames = stringKeynames.replace('\'','')
		stringKeynames = stringKeynames.replace('(','')
		stringKeynames = stringKeynames.replace(')','')
		self.batch = BatchStatement(consistency_level=ConsistencyLevel.QUORUM)

                if not type(self.dict_keynames) is tuple:
		    query+= stringKeynames + ", "
		else:
                    for i,k in enumerate(self.dict_keynames):
			query += self.dict_keynames[i] + ","

                if not type(self.dict_name) is tuple:
                    query += self.dict_name
		else:
                    lenv = len(self.dict_name) - 1
                    for i, v in enumerate(self.dict_name):
                        if i < lenv:
                            query += self.dict_name[i] + ","
                        else:
                            query += self.dict_name[i]

                query += ") VALUES ("

                if not type(self.dict_keynames) is tuple:
		    query += "?, "
		else:
                    for ind, k in enumerate(self.dict_keynames):
			query += "%(k" + str(ind) + ")s, "
                
                if not type(self.dict_name) is tuple:
                    query += "?)"
                else:
                    lenv = len(self.dict_name) - 1
                    for ind, v in enumerate(self.dict_name):
                        if ind < lenv:
			    query += "%(v" + str(ind) + ")s, "
                        else:
			    query += "%(v" + str(ind) + ")s)"

		self.insert_data = self.prepareQuery(query)

	        query1 = "self.batch.add(self.insert_data, ("
	        for k,v in self.dictCache.cache.iteritems():
		    if v[1] == 'Sent':
		        value = v[0]
		        if type(self.dict_keynames) == tuple:
			    lenk = len(k) - 1
                            query = query1
			    for ind,val in enumerate(k):
			        if ind < lenk:
			            if self.types[str(self.dict_keynames[ind])] == 'text':
				        query += "\'" + str(k[ind]) + "\', "
                                    else:
				        query += ""   + str(k[ind]) + ", "
			        else:
			            if self.types[str(self.dict_keynames[ind])] == 'text':
				        query += "\'" + str(k[ind]) + "\'"
                                    else:
				        query += ""   + str(k[ind]) + ""
		        else:
			    if self.types[str(self.dict_keynames)] == 'text':
			        query = query1 + "\'" + str(k) + "\'"
			    else:
			        query = query1 + ""   + str(k) + ""
                        
		        if type(value) == list:
			    newv = value
			    value = newv[0]
                        if type(self.dict_name) == tuple:
                            lenv = len(value) - 1
                            for ind, val in enumerate(value):
                                if ind < lenv:
                                    if self.types[str(self.dict_name[ind])] == 'text':
			                query += ", \'" + str(value[ind]) + "\'"
                                    else:
			                query += ", "   + str(value[ind]) + ""
                                else:
                                    if self.types[str(self.dict_name[ind])] == 'text':
			                query += ", \'" + str(value[ind]) + "\'))"
                                    else:
			                query += ", "   + str(value[ind]) +   "))"
                        else:
		            if self.types[str(self.dict_name)] == 'text':
			        query += ", \'" + str(value) + "\'))"
		            else:
			        query += ", "   + str(value) +   "))"
		        exec(query)
		        self.dictCache[k]=[value,'Sync']
	    self.dictCache.sents = 0

	    done = False
	    sessionExecute = 0
	    errors = 0
	    sleepTime = 0.5
	    while done == False:
		while sessionExecute < 10:
		    try:
			self.session.execute(self.batch)
			self.firstCounterBatch = 1
			self.firstNonCounterBatch = 1
			sessionExecute = 10
		    except Exception as e:
			if sessionExecute == 0:
			    print "sessionExecute Errors in SetItem----------------------------"
			errors = errors + 1
			time.sleep(sleepTime)
			if sleepTime < 40:
			    print "sleepTime:", sleepTime, " - Increasing"
			    sleepTime = sleepTime * 1.5 * random.uniform(1.0, 3.0)
			else:
			    print "sleepTime:", sleepTime, " - Decreasing"
			    sleepTime = sleepTime / 2
			if sessionExecute == 9:
			    print "tries:", sessionExecute
			    print "Error: Cannot execute query in setItem. Exception: ", e
			    print "GREPME: The queries were %s"%(query)
			    print "GREPME: The values were " + str(d)
			    raise e
			sessionExecute = sessionExecute + 1
		done = True
	    if errors > 0:
		if errors == 1:
		    print "Success after ", errors, " try"
		else:
		    print "Success after ", errors, " tries"
                errors = 0

    def __getitem__(self, key):
        start = time.time()
        if self.mypo.persistent == False:
            try:
                return dict.__getitem__(self, key)
            except Exception as e:
                return "Object " + str(self.dict_name) + " with key " + str(key) + " not in dict"
        else:
            self.reads += 1
            if self.use_cache == True:
                if len(self.dictCache.cache) >= max_cache_size:
                    print "cache full!"
                    self.writeitem()
                    self.dictCache.cache = {}
	        if key in self.dictCache.cache:
                    val = self.dictCache.cache[key]
                    if not val[1] == 'Requested': #Sync or Sent
                        print "not val[1] == Requested"
                        self.cache_hits += 1                                                                       #STATISTICS
			item = val[0]
                        end = time.time()                                                                          #STATISTICS
                        self.cache_hits_time += (end - start)                                                      #STATISTICS
                        if self.prefetch == True and ((self.reads % prefetch_size == 1) or self.reads == 1):
                            print "prefetch_data1"
                            self.prefetch_data(key, prefetch_size, prefetch_distance)
                            self.cache_hits_graph += "P"                                                           #STATISTICS
                        self.cache_hits_graph += "X"                                                               #STATISTICS
			return item
                    else:
                        print "val[1] == Requested"
                        status = ""
                        try:
                            if key in self.dictCache.pending_requests:
                                print "key in  self.dictCache.pending_requests"
                                if self.dictCache[key][0] == None:                                                
                                    print "self.dictCache[key][0] == None"
                                    self.pending_requests += 1                                                     #STATISTICS
                                    status = "PR"                                                                  #STATISTICS
                                    start2 = time.time()
                                    print "about to calculate results"                                             #HERE WE ARE
                                    print "self.dictCache.pending_requests:", self.dictCache.pending_requests
                                    print "key to search:", key
                                    print "self.dictCache.pending_requests[" + str(key) + "]:" + str(self.dictCache.pending_requests[key])
                                    print "self.dictCache.pending_requests[key].ready():     ", self.dictCache.pending_requests[key].ready()
                                    results = self.dictCache.pending_requests[key].get()
                                    print "results calculated"
                                    end2 = time.time()                                                             #STATISTICS
                                    self.pending_requests_time_res += (end2 - start2)                              #STATISTICS
                                else:                                   
                                    print "not self.dictCache[key][0] == None"
                                    self.cache_hits += 1                                                           #STATISTICS
                                    status = "CH"                                                                  #STATISTICS
                                del self.dictCache.pending_requests[key]
                        except Exception as e:
                            print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                            print "Exception:", e
                            print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                            self.cache_prefetchs_fails += 1                                                        #STATISTICS
		            item=self.readitem(key)
                    	    self.dictCache[key] = [item, 'Sync']
			    end = time.time()                                                                      #STATISTICS
                            self.pending_requests_fails_time += (end - start)                                      #STATISTICS
                            self.cache_hits_graph += "R"                                                           #STATISTICS
                            if self.prefetch == True and ((self.reads % prefetch_size == 1) or self.reads == 1):
                                print "prefetch_data2"
                                self.prefetch_data(key, prefetch_size, prefetch_distance)
                                self.cache_hits_graph += "P"                                                       #STATISTICS
			    return item
                        for entry in results:
                    	    self.dictCache[results[0]] = [results[1], 'Sync']
                        item = self.dictCache[key][0]
                        self.dictCache[key] = [item, 'Sync']
			end = time.time()                                                                          #STATISTICS
                        if status == "CH":                                                                         #STATISTICS
                            self.cache_hits_time += (end - start)                                                  #STATISTICS
                            self.cache_hits_graph += "X"                                                           #STATISTICS
                        else:                                                                                      #STATISTICS
                            self.pending_requests_time += (end - start)                                            #STATISTICS
                            self.cache_hits_graph += "R"                                                           #STATISTICS
                        if self.prefetch == True and ((self.reads % prefetch_size == 1) or self.reads == 1):
                            print "prefetch_data3"
                            self.prefetch_data(key, prefetch_size, prefetch_distance)
                            self.cache_hits_graph += "P"                                                           #STATISTICS
			return item
                else:
                    self.miss = self.miss + 1                                                                      #STATISTICS
                    try:
                        item = self.readitem(key)
	            except Exception as e:
		        raise KeyError
                    self.dictCache[key] = [item, 'Sync']
                    end = time.time()                                                                              #STATISTICS
                    self.miss_time += (end - start)                                                                #STATISTICS
                    if self.prefetch == True and ((self.reads % prefetch_size == 1) or self.reads == 1):
                        print "prefetch_data4"
                        self.prefetch_data(key, prefetch_size, prefetch_distance)
                        print "prefetch_data4done"
                        self.cache_hits_graph += "P"                                                               #STATISTICS
                    self.cache_hits_graph += "_"                                                                   #STATISTICS
		    return item
                return item
            else:
                self.miss += 1                                                                                     #STATISTICS
		item = self.readitem(key)
                end = time.time()                                                                                  #STATISTICS
                self.miss_time += (end - start)                                                                    #STATISTICS
                return item

    def readitem(self, key):
            query = "SELECT "
            if type(self.dict_name) == tuple:
                lenv = len(self.dict_name) - 1
                for ind, val in enumerate(self.dict_name):
                    if ind < lenv:
                        query += str(self.dict_name[ind]) + ", "
                    else:
                        query += str(self.dict_name[ind])
            else:
                query += str(self.dict_name)
            query += " FROM " + self.keyspace + ".\"" + self.mypo.name + "\" WHERE "
            if not type(key) is tuple:
                key = str(key).replace("[","")
                key = str(key).replace("]","")
                key = str(key).replace("'","")
                if self.types[str(self.dict_keynames)] == 'text':
                    query+=self.dict_keynames + " = \'" + str(key) + "\' LIMIT 1"
                else:
                    query+=self.dict_keynames + " = "   + str(key) +   " LIMIT 1"
            else:
            	lenk = len(key) - 1
            	for i, k in enumerate(key):
                    if i < lenk:
                        if self.types[str(self.dict_keynames[i])] == 'text':
                            query += self.dict_keynames[i] + " = \'" + str(k) + "\' AND "
                        else:
                            query += self.dict_keynames[i] + " = "   + str(k) +   " AND "
                    else:
                        if self.types[str(self.dict_keynames[i])] == 'text':
                            query += self.dict_keynames[i] + " = \'" + str(k) + "\' LIMIT 1;"
                        else:
                            query += self.dict_keynames[i] + " = "   + str(k) +   " LIMIT 1;"

            session = self.session
            errors = 0
            totalerrors = 0
            sleepTime = 0.5
            done = False
            while done == False:
                sessionExecute = 0
                while sessionExecute < 5:
                    try:
                        result = session.execute(query)
                        item = ''
                        for row in result:
                            lenr = len(row) - 1
                            for i, val in enumerate(row):
                                if i < lenr:
                                    item += str(val) + ", "
                                else:
                                    item += str(val)
                        sessionExecute = 5
                    except Exception as e:
                        if sessionExecute == 0:
                            print "sessionExecute Errors in GetItem----------------------------"
                            errors = 1
                        time.sleep(sleepTime)
                        if sleepTime < 3:
                            sleepTime = sleepTime * 1.5 * random.uniform(1.0, 3.0)
                            totalerrors = totalerrors + 1
                        else:
                            sleepTime = sleepTime / (totalerrors + 1)
                            sleepTime = sleepTime * totalerrors
                        if sessionExecute == 3:
                            session.shutdown()
                            time.sleep(10)
                            session = self.mypo.cluster.connect(self.keyspace)
                        if sessionExecute == 4:
                            print "tries:", sessionExecute
                            print "GREPME: The queries were %s"%(query) 
                            print "Error: Cannot execute query in getItem. Exception: ", e
                            raise e
                        sessionExecute = sessionExecute + 1
                done = True
            if len(result) == 0:
                print "len(result) = 0"
                raise KeyError
            if errors == 1:
                print "tries:", sessionExecute
                print "total errors:", totalerrors
            return item

    def prefetch_data(self, key, prefetch_size, prefetch_distance):
        self.prefetch_execs += 1                                                                                    #STATISTICS
        #self.prefetch = False     #testing
        keyPosition = self.blockKeys.index(int(key))

        init_pos = keyPosition + prefetch_distance
        if init_pos > len(self.blockKeys):
            init_pos = len(self.blockKeys) - 1

        end_pos = init_pos + prefetch_size
        if end_pos > len(self.blockKeys):
            end_pos = len(self.blockKeys) - 1

        
        query = "SELECT * FROM " + self.keyspace + ".\"" + self.mypo.name + "\" WHERE "
        if not type (key) is tuple:
	        query+=self.dict_keynames + " = " + "?" #removed ; and removed LIMIT 1
	else:
		lenk = len(key)
		for ind, k in enumerate(key):
			if ind < lenk:
				query += self.dict_keynames[ind] + " = ? AND"
			else:
				query += self.dict_keynames[ind] + " = ?" #removed ; and removed LIMIT 1
        nmaps=4
        size_per_map = (end_pos-init_pos)/nmaps
        start_ix     = init_pos
	end_ix       = init_pos + size_per_map
        for i in range(0,nmaps):
            # chunksize, concurrency, query, arguments
            keyForMap = {}
            for ix in range(start_ix, end_ix):
                target = self.blockKeys[ix]
                self.cache_prefetchs += 1                                                                    #STATISTICS
                if not target in self.dictCache.cache:
                    keyForMap[ix]=target
                    self.dictCache[target] = [None,'Requested']
	
            parameters = [(x,) for x in keyForMap.itervalues()]
            params = (1, 2, query, parameters)
            res = self.prefetchManager.get_results(params)
            '''
            for ind,val in enumerate(keyForMap.itervalues()):
                print "ind:", ind
                print "val:", val
                try:
                    self.dictCache.pending_requests[val] = res.pop(0)
                except Exception as e:
                    print "exception:", e
            '''
            ind = start_ix
            for val in res:
                print "ind:", ind
                print "val:", val
                try:
                    self.dictCache.pending_requests[keyForMap[ind]] = val
                except Exception as e:
                    print "exception:", e
                ind += 1
	    start_ix = start_ix+size_per_map
	    end_ix = min(end_ix+size_per_map, end_pos)

    def len(self):
        keyspace = 'config' + execution_name
        try:
            session = self.mypo.cluster.connect(execution_name)
        except Exception as e:
            pass
        query = "SELECT count(*) FROM " + execution_name + ".\"" + self.mypo.name + "\";"
	done = False
        while done == False:
            try:
                result = session.execute(query)
                item = 0
                for row in result:
                    item = row[0]
            except Exception as e:
                pass
            finally:
                session.shutdown()
            done = True
        return item

    def keys(self):
        if self.mypo.persistent == False:
            return dict.keys(self)
        else:
            return PersistentKeyList(self)

    def sortedkeys(self):
        if self.mypo.persistent == False:
            return sorted(dict.keys(self))
        else:
            return PersistentKeyList(self)

    # Make PersistentDict serializable with pickle 
    def __getstate__(self):
        return (self.mypo, self.dict_name, self.dict_keynames, dict(self))
    def __setstate__(self, state):
        self.mypo, self.dict_name, self.dict_keynames, data = state
        self.update(data)  # will *not* call __setitem__
    def __reduce__(self):
        return (PersistentDict, (), self.__getstate__())

class dictContext():
    def __init__(self, storageObj):
	self.storageObj = storageObj
    
    def __enter__(self):
        if self.storageObj.__class__.__name__ == 'PersistentDict':
            self.storageObj.batchvar = 'true'
            #NEW LINE
            #self.storageObj.dictCache.valtype = self.storageObj.types[self.dict_name]
        else:
            keys = self.storageObj.keyList[self.storageObj.__class__.__name__]
            exec("self.storageObj." + str(keys[0]) + ".batchvar  = 'true'")
            #NEW LINE
            #exec("self.storageObj." + str(keys[0]) + ".dictCache.valtype = self.storageObj." + str(keys[0]) + ".types[str(self.storageObj." + str(keys[0]) + ".dict_name)]")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.storageObj.__class__.__name__ == 'PersistentDict':
	    #self.prefetchManager.close_pool()
	    midict                = self.storageObj
	    midict.batchvar       = 'false'
            if midict.use_cache == True:
	        micache               = midict.dictCache
            valType               = self.storageObj.types[str(self.dict_name)]
            keyType               = self.storageObj.types[str(self.dict_keynames)]
        else: 
	    #exec("self.storageObj." + str(keys[0]) + ".prefetchManager.close_pool()")
            keys = self.storageObj.keyList[self.storageObj.__class__.__name__]
            exec("midict          = self.storageObj." + str(keys[0]))
            exec("midict.batchvar = 'false'")
            if midict.use_cache == True:
                exec("micache         = midict.dictCache")
            exec("valType         = self.storageObj." + str(keys[0]) + ".types[str(self.storageObj." + str(keys[0]) + ".dict_name)]")
            exec("keyType         = self.storageObj." + str(keys[0]) + ".types[str(self.storageObj." + str(keys[0]) + ".dict_keynames)]")
        if midict.use_cache == True:
            midict.syncs = midict.syncs + micache.sents
	    midict.writeitem()
        else:
            midict.syncs = midict.syncs + midict.batchCount
            midict.session.execute(midict.batch)
            midict.batchvar = False
