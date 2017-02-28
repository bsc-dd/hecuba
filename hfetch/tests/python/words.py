import sys
import gc
import objgraph
import time
sys.path.append(".")
import hfetch

hfetch.connectCassandra(["minerva-5"],9042)
a = hfetch.Hcache(10,"wordcount","words","WHERE token(position)>=? AND token(position)<?;",[(-9070430489100700000,-8900450532247928832),
#(-8670430489100700000,-8400450532247928832),
#(-8370430489100700000,-8100450532247928832),
(-8000430489100700000,-7950450532247928832)]
,["position"],["wordinfo"])




print "Iterate"
itera = a.iteritems(1)


while True:
    try:
        data = itera.get_next()
    except StopIteration:
        break
        
wait = raw_input("End test?")
data = None
itera = None
objgraph.show_most_common_types()
print 'end it'


#track.print_diff()
wait = raw_input("End test?")

a = None


gc.collect()
wait = raw_input("End test?")


