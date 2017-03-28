from hfetch import *
import numpy as np
connectCassandra(["127.0.0.1"],9042)

elem_row=2048
txt_elem_row = str(elem_row)

a = Hcache("test","arrays","WHERE token(partid)>=? AND token(partid)<?;",[(-8070430489100700000,8070450532247928832)],["partid"],[{"name":"image_block","type":"double","dims":txt_elem_row+'x'+txt_elem_row,"partition":"true"},"image_block_pos"],{})


bigarr=np.arange(pow(elem_row,2)).reshape(elem_row,elem_row)
bigarr.itemset(0,14.0)
print 'Array to be written', bigarr.astype('d')
import time
t1 = time.time()
a.put_row([300],[bigarr.astype('d')])
time.sleep(3)
#othw we ask for the row before it has been processed
result = a.get_row([300])
print 'Written:', bigarr.astype('d')
resarr = result[0]
print "And the result is... ", resarr.reshape((2048,2048))
print 'Elapsed time', time.time()-t1
print '2D, elem dimension: ', elem_row
    

