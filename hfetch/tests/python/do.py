import sys
sys.path.append(".")
from hfetch import *
connectCassandra(["127.0.0.1"],9042)
a = Hcache("test","particle","WHERE token(partid)>=? AND token(partid)<?;",[(-8070430489100700000,8070450532247928832)],["partid","time"],["x"],{'cache_size':'10','writer_buffer':20})
itera = a.iteritems(123)
print itera.get_next()
wait = raw_input("End test?")
