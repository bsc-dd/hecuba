import time
from hfetch import *

contact_names = 'minerva-5'
nodePort = 9042
keyspace = 'case18'
table = 'particle'
query_get = 'SELECT * FROM case18.particle WHERE partid=? AND time=? ;'
query_token ='SELECT * FROM case18.particle WHERE token(partid)>? AND token(partid)<? ;'

token_ranges = [(8070430489100699999,8070450532247928832)]
num_keys = 1024
size = 1024

success = connectCassandra(nodePort, contact_names)
assert (success)

cache = hcache(size,table,keyspace,query_get,[(8070430489100699999,8070450532247928832)],100,query_token)


#clustering key
ck = float(0.003)
t1 = time.time()
for pk in xrange(1, num_keys + 1):
    try:
        result = cache.get_row([pk, ck])
    except Exception as e:
        print "Error when retrieving value from cache:", e
print 'items in res: ', result
print 'time - load C++ cache with cassandra data: ', time.time() - t1

t1 = time.time()
for pk in xrange(1, num_keys + 1):
    try:
        result = cache.get_row([pk, ck])
    except Exception as e:
        print "Error when retrieving value from cache:", e
# print 'items in res: ',len(result)
print 'time - read data from C++ cache: ', time.time() - t1

py_dict = {}

t1 = time.time()
for pk in xrange(1, num_keys + 1):
    try:
        py_dict[pk] = cache.get_row([pk, ck])
    except Exception as e:
        print "Error when retrieving value from cache:", e
print 'time - load data into python dict: ', time.time() - t1
# print 'size ', len(py_dict)
# print 'items in res: ',len(py_dict[1])

t1 = time.time()
for pk in xrange(1, num_keys + 1):
    try:
        result = py_dict[pk]
    except Exception as e:
        print "Error when retrieving value from cache:", e
print 'time - read data from the python dict: ', time.time() - t1
# print 'size ', len(py_dict)
# print 'items in res: ',len(py_dict[1])
