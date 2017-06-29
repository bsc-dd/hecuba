from hfetch import *

connectCassandra(["minerva-5"],9042)
nparts=6000000
p=1000
t_f=0
t_t=5764607523034234880
tkn_size=(t_t-t_f)/(nparts/p)
tkns=[(a,a+tkn_size) for a in xrange(t_f,t_t-tkn_size,tkn_size)]
a = Hcache(10,"case18","particle","WHERE token(partid)>=? AND token(partid)<?;",tkns,["partid","time"],["x"])
def readAll(iter):
    count = 1
    i=iter.get_next()
    while ( i is not None):
        i=iter.get_next()
        count +=1
	if count%100000==0:
            print count
    print "iter has %d elements"%count
 
from time import time

start=time()

readAll(a.iterkeys(100))
print "finshed into %d"%(time()-start)
a = None
disconnectCassandra()


