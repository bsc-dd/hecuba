import time
from hfetch import *

contact_names = ['127.0.0.1']

nodePort = 9042
keyspace = 'test'
table = 'particle'

token_ranges = [(8070430489100699999,8070450532247928832)]
num_keys = 10001
size = 10001

try:
    connectCassandra(contact_names,nodePort)
except Exception:
    print 'can\'t connect, verify the contact poins and port',contact_names,nodePort

cache = Hcache(keyspace,table,"",token_ranges,["partid","time"],["ciao","x","y","z"],{'cache_size':size})


result=None
#clustering key
t1 = time.time()
for pk in xrange(0, num_keys):
    ck = pk*10
    try:
        result = cache.get_row([pk, ck])
    except KeyError as e:
        print "Error when retrieving value from cache:", e, [pk,ck]
print 'items in res: ', result
print 'time - load C++ cache with cassandra data: ', time.time() - t1

t1 = time.time()
for pk in xrange(0, num_keys):
    ck = pk*10
    try:
        result = cache.get_row([pk, ck])
    except KeyError as e:
        print "Error when retrieving value from cache:", e, [pk,ck]
# print 'items in res: ',len(result)
print 'time - read data from C++ cache: ', time.time() - t1

py_dict = {}
cache = Hcache(keyspace,table,"",[(8070430489100699999,8070450532247928832)],["partid","time"],["ciao","x","y","z"],{'cache_size':num_keys})

t1 = time.time()
for pk in xrange(0, num_keys):
    ck = pk*10
    try:
        py_dict[pk] = cache.get_row([pk, ck])
    except KeyError as e:
        print "Error when retrieving value from cache:", e, [pk,ck]
print 'time - load data into python dict: ', time.time() - t1
# print 'size ', len(py_dict)
# print 'items in res: ',len(py_dict[1])

t1 = time.time()
for pk in xrange(0, num_keys):
    try:
        result = py_dict[pk]
    except KeyError as e:
        print "Error when retrieving value from cache:", e, [pk,ck]
print 'time - read data from the python dict: ', time.time() - t1
# print 'size ', len(py_dict)
# print 'items in res: ',len(py_dict[1])
