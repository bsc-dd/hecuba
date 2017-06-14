from cassandra.cluster import *
import numpy as np
'''''''''
Retrieves bytes and checks they have been written correctly
Note: Rebuilds the array assuming the retrieved subarrays are ordered
'''''''''

size = 5*5*5 #2048*2048
data_type = np.int32 #np.double
shape = (5,5,5)

cluster = Cluster(['127.0.0.1'])
session = cluster.connect('test')
prepared = session.prepare('SELECT * FROM test.arrays;')
result = session.execute(prepared)
list = [(a.image_block_pos,a.image_block) for a in result ]
text=''
for i in xrange(0,1):
    text=text+list[i][1]
    if i!=list[i][0]:
        print 'Wrong i',i
    
print 'bytes length',len(text)
bigarr = np.fromstring(text,dtype=data_type, count=size)

reshaped = bigarr.reshape(shape)
print reshaped
