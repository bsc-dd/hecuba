from hfetch import *
import numpy as np
connectCassandra(["127.0.0.1"],9042)


elem_row=512
txt_elem_row = str(elem_row)



a = Hcache("test","arrays","WHERE token(partid)>=? AND token(partid)<?;",[(-8070430489100700000,8070450532247928832)],["partid"],[{"name":"image_block","type":"double","dims":txt_elem_row+'x'+txt_elem_row}],{})



bigarr=np.arange(pow(elem_row,2)).reshape(elem_row,elem_row)

import time
t1 = time.time()
a.put_row([300],[bigarr.astype('d')])
print 'Elapsed time', time.time()-t1
print '2D, elem dimension: ', elem_row
    

