from hfetch import *

'''''''''
This test iterates over a small amount of data
'''''''''
connectCassandra(["127.0.0.1"], 9042)

# this should fail since a key can not be a column name at the same time (key=time, column=time)
a = Hcache("test", "particle", "WHERE token(partid)>=? AND token(partid)<?;",
           [(-8070430489100700000, 8070450532247928832)], ["partid", "time"], ["time", "x"],
           {'cache_size': '10', 'writer_buffer': 20})

# now this should work
a = Hcache("test", "particle", "WHERE token(partid)>=? AND token(partid)<?;",
           [(-8070430489100700000, 8070450532247928832)], ["partid", "time"], ["x"],
           {'cache_size': '10', 'writer_buffer': 20})

to = a.iterkeys(10000)

res = 1

while res is not None:
    try:
        res = to.get_next()
    except StopIteration:
        break
    print str(res)
print 'done'
