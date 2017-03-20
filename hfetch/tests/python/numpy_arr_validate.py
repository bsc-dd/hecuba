# coding: utf-8

from cassandra.cluster import *
import numpy as np

cluster = Cluster(['127.0.0.1'])
session = cluster.connect('test')
prepared = session.prepare('SELECT * FROM test.arrays;')
result = session.execute(prepared)
list = [(a.image_block_pos,a.image_block) for a in result ]
text=''
for i in xrange(0,6):
    text=text+list[i][1]
    
print 'bytes length',len(text)
print np.fromstring(text,dtype=np.double, count=2048)
