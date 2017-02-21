from hfetch import *

succeeds = connectCassandra(['minerva-5'],9042)
print 'Connection succeeds', succeeds
assert (succeeds)

token_ranges=[(8070400480100699999,8070430489100699999),(8070430489100700000,8070450532247928832)]
table = Hcache(20,'case18','particle',"WHERE token(partid)>=? AND token(partid)<?;",[(8070430489100700000,8070450532247928832)],["partid","time"],["time","x"])


q1 = table.get_row([433,float(0.003)])
print q1
print table.get_row([433,float(0.003)])
print table.get_row([133,float(0.001)])
print table.get_row([433,float(0.002)])
q2 = table.get_row([433,float(0.003)])
print q2

assert (q1 == q2)


disconnectCassandra()
