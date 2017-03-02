from hfetch import *
import numpy as np
connectCassandra(["127.0.0.1"],9042)


dims=3
data_len=5

a = Hcache(10,"test","bytes","WHERE token(partid)>=? AND token(partid)<?;",[(-8070430489100700000,8070450532247928832)],["partid"],[{"name":"data","type":"double","dims":"5x5x5"}])

bigarr=np.arange(pow(data_len,dims)).reshape(5,5,5)


a.put_row([100],[bigarr.astype('d')])
print a.get_row([100])[0]
