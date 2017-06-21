from hfetch import *
import time


contact_names = ['127.0.0.1']
nodePort = 9042
keyspace = 'test'
table = 'particle'

token_ranges = [(8070430489100699999,8070450532247928832)]

num_keys = 10001
non_existent_keys = 10

cache_size = num_keys+non_existent_keys

try:
    connectCassandra(contact_names,nodePort)
except Exception:
    print 'can\'t connect, verify the contact poins and port',contact_names,nodePort

cache = Hcache(keyspace,table,"",token_ranges,["partid","time"],["ciao","x","y","z"],{'cache_size':cache_size})

# Access the cache, which is empty and queries cassandra to retrieve the data
t1 = time.time()
error_counter = 0
for pk in xrange(0, num_keys+non_existent_keys):
    ck = pk*10
    try:
        result = cache.get_row([pk, ck])
    except KeyError as e:
        error_counter = error_counter + 1

print 'Retrieved {0} keys in {1} seconds. {2} keys weren\'t found, {3} keys weren\'t supposed to be found'.format(unicode(str(num_keys),'utf-8'),
    unicode(str(time.time()-t1),'utf-8'),unicode(str(error_counter),'utf-8'),unicode(str(non_existent_keys),'utf-8'))

assert(error_counter==non_existent_keys)

# Access the cache, which has already all the data and will ask cassandra only if
# the keys asked are not present
t1 = time.time()
error_counter = 0
for pk in xrange(0, num_keys+non_existent_keys):
    ck = pk*10
    try:
        result = cache.get_row([pk, ck])
    except KeyError as e:
        error_counter = error_counter + 1

print 'Retrieved {0} keys in {1} seconds. {2} keys weren\'t found, {3} keys weren\'t supposed to be found'.format(unicode(str(num_keys),'utf-8'),
    unicode(str(time.time()-t1),'utf-8'),unicode(str(error_counter),'utf-8'),unicode(str(non_existent_keys),'utf-8'))
assert(error_counter==non_existent_keys)
