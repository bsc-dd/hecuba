import os
import sys
import time

from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
from cassandra.query import tuple_factory
from cassandra.concurrent import *
from multiprocessing import Pool
import multiprocessing, logging
import itertools

def _multiprocess_get(params):
    print "multiprocessGet"
    return PrefetchManager._results_from_concurrent(params)

class PrefetchManager(object):

    pool = ''

    def __init__(self, cluster, process_count=None):
        
        logger = multiprocessing.log_to_stderr()
        logger.setLevel(multiprocessing.SUBDEBUG)
        self.pool = Pool(processes=process_count, initializer=self._setup, initargs=(cluster,))

    @classmethod
    def _setup(cls, cluster):
        cls.session = cluster.connect()

    def close_pool(self):
        self.pool.close()
        self.pool.join()

    def get_results(self, params):
        print"-------------------------------------"
        chunk_size =  params[0]
        concurrency = params[1]
        query =       params[2]
        arguments =   params[3]
        print "concurrency:", concurrency
        print "query:      ", query
        print "arguments:  ", arguments
        #print "                           chunk_size :         ", chunk_size
        #print "type(chunk_size): ", type(chunk_size)
        #print "type(concurrency):", type(concurrency)
        #print "type(query):      ", type(query)
        #print "type(arguments):  ", type(arguments)
        #print "          len(arguments)              :         ", len(arguments)
        lenDivChs =     (len(arguments) / chunk_size)
        #print "         (len(arguments) / chunk_size):         ", lenDivChs
        if lenDivChs < 1:
	    lenDivChs = 1
        #print "     max((len(arguments) / chunk_size),1):      ", lenDivChs
        a = [query]* (lenDivChs)        #max((len(arguments) / chunk_size),1)
        b = [concurrency] * (lenDivChs) #max((len(arguments) / chunk_size),1)
        final_pars = list(zip(a,b,(arguments[n:n + chunk_size] for n in xrange(0, len(arguments), chunk_size))))
        if not len(arguments) % chunk_size == 0:
            final_pars.append((query,concurrency,arguments[len(arguments)-(len(arguments)%chunk_size):len(arguments)]))
        print "final_pars:", final_pars
        #r = self.pool.map_async(_multiprocess_get, final_pars)
        #r = map(_multiprocess_get, final_pars)
        r = []
        for value in final_pars:
            r.append(self.pool.apply_async(_multiprocess_get, [value]))
        return r

    @classmethod
    def _results_from_concurrent(cls, params):
        f = open('/home/galomar/.COMPSs/resultsconcurrent.txt', 'a')
        toWrite = ""
        print "_results_from_concurrent......................"
        query =       params[0]
        concurrency = params[1]
        arguments =   params[2]
        toWrite += 'params\n'
        toWrite += str(params)+'\n'
        f.write(toWrite)
        f.close()
        f = open('/home/galomar/.COMPSs/resultsconcurrent.txt', 'a')
        cls.prepared = cls.session.prepare(query)
        results = execute_concurrent_with_args(cls.session, cls.prepared, arguments,concurrency)
        toWrite += 'len(results)\n'
        toWrite += str(len(results))+'\n'
        print "len(results):", len(results)
        values = []
        i=0
        for (success,result) in results:
            if success:
                values.append(result[0])
                if i == 0:
                    print "result[0]:", result[0]
                i += 1
            else:
                print error
        f.write(toWrite)
        return values
