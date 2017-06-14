from hfetch import *
'''''''''
This test iterates over huge lines of text
'''''''''
connectCassandra(["minerva-5"], 19042)
itera = HIterator("wordcount", "fivegb",
           [(-9070430489100700000, -9000030489100700000),(-8070430489100700000, -8000030489100700000),(-7070430489100700000, -7000030489100700000)], ["position"], [],
           {'prefetch_size': '100', 'writer_buffer': 20})

data= None
while True:
    try:
        data = itera.get_next()
    except StopIteration:
        break

print data
