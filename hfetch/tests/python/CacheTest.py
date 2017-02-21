
import threading
import sys
import itertools
from hfetch import *

class PersistentDictCache(dict):
    pending_requests = {}
    cache = None

    def __missing__(self, key):
        print 'error: missing triggered'
        assert (False)
        return None

    def __init__(self, size, keyspace, table, query):
        super(dict,self).__init__()
        self.pending_requests = {}
        success = self.cache.connectCassandra(keyspace, nodePort, contact_names)
        assert(success)
        self.cache = hcache(size, table, keyspace, query)

    def __setitem__(self, key, value):
        success = self.cache.put_row([key,value])
        assert (success==0)

    def __getitem__(self, key):
        value = []
        try:
            value = self.cache.get_row(key)
        except Exception as e:
            print "Error when retrieving value from cache:", e
            print "self.cache:", self.cache
        return value

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cache.disconnectCassandra()


def call(api, part_keys, clust_keys):
    global finished
    for pk in part_keys:
        for ck in clust_keys:
            result = api[[pk,ck]]
            #print result
        # next returns value and increments subsequently
    if count.next() == max_parallelism-1:
        finished = True


contact_names = 'minerva-5'
nodePort = 9042
keyspace = 'case18'
table = 'particle'
query = 'SELECT * FROM case18.particle WHERE partid=? AND time=? ;'

count = itertools.count() #starts at 0
finished = False
max_parallelism = int(sys.argv[1])

num_keys=2048
cache_size = 2048
mydict = PersistentDictCache(cache_size, keyspace, table, query)

ths = []
k_to_thread=num_keys/max_parallelism
for i in xrange(0,num_keys,k_to_thread):
    ths.append(threading.Thread(target=call,kwargs = {'api':mydict, 'part_keys':list(range(i+1,i+k_to_thread+1)),'clust_keys':[float(0.003)]}))
for th in ths:
    th.start()
while not finished:
    pass
for th in ths:
    th.join()