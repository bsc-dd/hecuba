from hfetch import *
import numpy as np
connectCassandra(["127.0.0.1"],9042)
a = Hcache(10,"test","bytes","WHERE token(partid)>=? AND token(partid)<?;",[(-8070430489100700000,8070450532247928832)],["partid"],[{"name":"data","type":"float","dims":"64x64"}])

bigarr=np.array([[d*0.1 for d in xrange(64)] for i in ([x for x in xrange(64)])],np.float)

#print bigarr

a.put_row([100],[bigarr.astype('f')])
print a.get_row([100])[0].astype('f')
