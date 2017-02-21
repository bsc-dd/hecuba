import sys
import time
sys.path.append(".")
from hfetch import *
connectCassandra(["minerva-5"],9042)
a = Hcache(10,"wordcount","words","WHERE token(position)>=? AND token(position)<?;",[(-9070430489100700000,-5070450532247928832)],["position"],["wordinfo"])
time.sleep(5)
itera = a.iteritems(123)
print itera.get_next()
wait = raw_input("End test?")
