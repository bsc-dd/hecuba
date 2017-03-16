from hfetch import *

'''''''''
Simple test to store text and retrieves it
'''''''''
connectCassandra(["127.0.0.1"], 9042)
# CREATE TABLE test.bulk(partid int PRIMARY KEY, data text);
a = Hcache("test", "bulk", "WHERE token(partid)>=? AND token(partid)<?;", [(-8070430489100700000, 8070450532247928832)],
           ["partid"], ["data"], {'cache_size': '10', 'writer_buffer': 20})
i = 0
while i < pow(10, 3):
    a.put_row([i], ['someRandomText'])
    i += 1

itera = a.iteritems(10)

print itera.get_next()
wait = raw_input("End test?")
