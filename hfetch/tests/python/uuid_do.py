from hfetch import *
import uuid
'''''''''
Simple test to store text and retrieves it
'''''''''
connectCassandra(["127.0.0.1"], 9042)
# CREATE TABLE test.bulk(partid int PRIMARY KEY, data text);
a = Hcache("test", "uuid", "WHERE token(partid)>=? AND token(partid)<?;", [(-8070430489100700000, 8070450532247928832)],
           ["partid"], ["data"], {'cache_size': '10', 'writer_buffer': 20})

i = 0
while i < 1:# pow(10, 3):
    u = uuid.UUID('81da81e8-1914-11e7-908d-ecf4bb4c66c4') 
    print 'the uuid is ', u
    a.put_row([u], [i])
    i += 1
print 'done insert'
import time
time.sleep(1)
itera = a.iteritems(10)

L = uuid.UUID(itera.get_next()[0])
print 'got ', L
wait = raw_input("End test?")
