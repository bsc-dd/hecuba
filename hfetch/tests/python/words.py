from hfetch import *
'''''''''
This test iterates over huge lines of text
'''''''''
connectCassandra(["minerva-5"], 19042)
a = Hcache("wordcount", "words", "WHERE token(position)>=? AND token(position)<?;",
           [(-9070430489100700000, -9070030489100700000)], ["position"], ["wordinfo"],
           {'cache_size': '10', 'writer_buffer': 20})

print 'Prefetch starts'
itera = a.iteritems(123)

first = itera.get_next()
print 'First results ready'

wait = raw_input("Press to iterate over results")

while True:
    try:
        data = itera.get_next()
    except StopIteration:
        break

wait = raw_input("End test and write one retrieved line?")

print first
