from hfetch import *
import numpy as np
connectCassandra(["127.0.0.1"],9042)

succeeds = connectCassandra(['127.0.0.1'], 9042)
bigarr=np.array([[d*0.1 for d in xrange(64)] for i in ([x for x in xrange(64)])],np.float)

token_ranges = [(8070400480100699999, 8070430489100699999), (8070430489100700000, 8070450532247928832)]
# empty configuration parameter (the last dictionary) means to use the default config
token_range = [(-8070430489100700000,8070450532247928832)]

table = Hcache("test","bytes","WHERE token(partid)>=? AND token(partid)<?;", token_ranges, ["partid"],[{"name":"data","type":"float","dims":"64x64"}], {})

a.put_row([100],[bigarr.astype('f')])
print a.get_row([100])[0].astype('f')
