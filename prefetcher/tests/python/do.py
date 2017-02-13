import sys
sys.path.append(".")
from hfetch import *
connectCassandra(["minerva-5"],9042)
a = Hcache(10,"case18","particle","WHERE token(partid)>=? AND token(partid)<?;",[(8070430489100700000,8070450532247928832)],["partid","time"],["time","x"])
itera = a.iteritems(123)
print itera.get_next()
wait = raw_input("End test?")
