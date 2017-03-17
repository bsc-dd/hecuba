import sys
sys.path.append(".")
from hfetch import *
connectCassandra(["127.0.0.1"],9042)
import numpy as np
import marshal
array = np.array([[3212,21321],[986778,1241]])
a = Hcache(10,"test","bytes","WHERE token(partid)>=? AND token(partid)<?;",[(-8070430489100700000,8070450532247928832)],["partid"],["data"])
#print 'should be: ', str(array.tostring().encode('hex'))
#a.put_row([500],[array])

'''''''
itera = a.iteritems(2)
print 'result '
while True:
    try:
        print itera.get_next()
    except StopIteration:
        break
'''''''

wait = raw_input("End test?")
