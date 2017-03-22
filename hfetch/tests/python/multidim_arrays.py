from hfetch import *
import numpy as np
connectCassandra(["127.0.0.1"],9042)

import time
dims=3
data_len=5

a = Hcache("test","arrays","WHERE token(partid)>=? AND token(partid)<?;",[(-8070430489100700000,8070450532247928832)],["partid"],[{"name":"image_block","type":"int","dims":"5x5x5"},"image_block_pos"],{})

bigarr=np.arange(pow(data_len,dims)).reshape(5,5,5)


a.put_row([100],[bigarr.astype('i')])
#othw we ask for the row before it has been processed
time.sleep(2)

result = a.get_row([100])
print result
