def connectCassandra(ep, port):
    print 'asking hfetch to connect to %s,%s' % (ep, port)


class Hcache:
    def __init__(self, max_cache_size, keyspace, table, token_range_p,
                 tokens, primary_keys, column_names):
        print 'created hcahce with %s,%s,%s,%s,%s,%s,%s' % (max_cache_size, keyspace, table, token_range_p,
                                                            tokens, primary_keys, column_names)

    def get_row(self, key):
        print "asking hcache to get key=", str(key)

    def put_row(self, all):
        print "asking hcache to put all=", str(all)

    def iterterms(self):
        return FakeIter()

    def iterkeys(self):
        return FakeIter()

    def itervalues(self):
        return FakeIter()


class FakeIter:
    def get_next(self):
        print "asking get next, but nothing here"
        return None
