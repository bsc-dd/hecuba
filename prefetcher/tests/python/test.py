from hfetch import *

succeeds = connectCassandra(9042,'minerva-5')
print 'Connection succeeds', succeeds
assert (succeeds)


token_ranges=[(8070400480100699999,8070430489100699999),(8070430489100700000,8070450532247928832)]
table = hcache(20,'particle','case18','SELECT * FROM case18.particle WHERE partid=? AND time=? ;',token_ranges,100,'SELECT * FROM case18.particle WHERE token(partid)>? AND token(partid)<? ;')


q1 = table.get_row([433,float(0.003)])
print q1
print table.get_row([433,float(0.003)])
print table.get_row([133,float(0.001)])
print table.get_row([433,float(0.002)])
q2 = table.get_row([433,float(0.003)])
print q2

assert (q1 == q2)

q3 = table.get_next()

print 'Get next: ', q3
print 'Get next coherency test', table.get_row([q3[0],q3[1]])==q3


myrow = [123, 0.003000000026077032, 3.1482e-318, 10, 1, 1.81395e-315, 22, 0.08470910042524338, 0.07382529973983765, 0.00011495799844851717, 0.024705199524760246, 0.0746655985713005, 0.00011649299995042384, -0.02430409938097, 0.3032679855823517, 0.00047390200779773295]

table.put_row(myrow)
print 'put row successful and coherent: ', (table.get_row([123,float(0.003)])==myrow)


disconnectCassandra()
