from hfetch import *

succeeds = connectCassandra(['127.0.0.1'], 9042)

token_ranges = [(8070400480100699999, 8070430489100699999), (8070430489100700000, 8070450532247928832)]
# empty configuration parameter (the last dictionary) means to use the default config
table = Hcache('test', 'particle', "WHERE token(partid)>=? AND token(partid)<?;", token_ranges, ["partid", "time"],
               ["x", "y", "z"], {})


def get_data(t, keys):
    data = None
    try:
        data = t.get_row(keys)
    except KeyError:
        print 'not found'
    return data


q1 = get_data(table, [433, 4330])# float(0.003)
print 'q1: ',q1
print get_data(table, [133, 1330])
print get_data(table, [433, 4330])
q2 = get_data(table, [433, 4330])
print 'q2: ',q2
assert (q1 == q2)
print 'Assert q1 and q2 are equals succeeds'