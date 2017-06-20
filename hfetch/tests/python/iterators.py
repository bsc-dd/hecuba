from time import time
from hfetch import *

'''''''''
This test iterates over a huge amount of data
'''''''''
connectCassandra(["127.0.0.1"], 9042)
tkns=[(pow(-2,61),pow(2,61))]
a = Hcache("test", "words", "WHERE token(position)>=? AND token(position)<?;", tkns, ["position"], ["wordinfo"],
           {'cache_size': '100', 'writer_buffer': 20})



start = time()
myIter = a.iteritems({"prefetch_size":100,"update_cache":"yes"})

data = []
for i in xrange(0,10):
    data.append(myIter.get_next())

assert(len(data)>0)
first_data = data[0]

assert (len(first_data)==2)
first_key = [first_data[0]]

assert (type(first_key[0])==int)
somedata= a.get_row(first_key)

assert ((first_key+somedata) == first_data)

count = len(data)
i = []
while (i is not None):
    try:
        i = myIter.get_next()
    except StopIteration:
        print 'End of data, items read: ', count, ' with value ', i
        break
    count = count + 1

print 'data was: \n', data
