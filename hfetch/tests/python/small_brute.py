from hfetch import  *


connectCassandra(["minerva-5"],9042)                                 

a = Hcache(10,"case18","particle","WHERE token(partid)>=? AND token(partid)<?;",[(8070430489100700000,8070450532247928832)],["partid","time"],["time","x"])               


to=a.iterkeys(10000)

res=1

while(res is not None):
     res=to.get_next()
     print str(res)
print 'done'
