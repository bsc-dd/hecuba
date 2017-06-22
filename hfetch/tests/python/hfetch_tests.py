import unittest
import time

from hecuba import config



class Hfetch_Tests(unittest.TestCase):
    contact_names = ['127.0.0.1']
    nodePort = 9042

    @staticmethod
    def setUpClass():
        config.reset(mock_cassandra=False)
        config.session.execute(
            "CREATE KEYSPACE IF NOT EXISTS test WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")
        time.sleep(10)


    def test_iterate_brute(self):
        from hfetch import connectCassandra
        from hfetch import Hcache

        '''''''''
        This test iterates over a huge amount of data
        '''''''''

        keyspace = "test"
        table = "particle"
        nparts = 10000  # Num particles in range

        config.session.execute("DROP TABLE IF EXISTS %s.%s;" %(keyspace, table))
        config.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid int, time float, ciao text,"
                               "x float, y float, z float, PRIMARY KEY(partid,time));"% (keyspace,table))

        for i in xrange(0,nparts):
            vals=','.join(str(e) for e in [i,i/.1,i/.2,i/.3,i/.4,"'"+str(i*60)+"'"])
            config.session.execute("INSERT INTO %s.%s(partid , time , x, y , z,ciao ) VALUES (%s)"% (keyspace,table,vals))

        try:
            connectCassandra(self.contact_names, self.nodePort)
        except RuntimeError, e:
            print e
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort

        p = 100  # Num partitions

        t_f = pow(-2,63)  # Token begin range
        t_t = pow(2,63)
        # Token blocks
        tkn_size = (t_t - t_f) / (nparts / p)
        tkns = [(a, a + tkn_size) for a in xrange(t_f, t_t - tkn_size, tkn_size)]

        keys = ["partid", "time"]
        values = ["x"]

        hcache_config = {'cache_size': '100', 'writer_buffer': 20}

        token_query = "WHERE token(partid)>=? AND token(partid)<?;"



        cache = Hcache(keyspace,table, token_query , tkns, keys, values, hcache_config)

        hiter_config = {"prefetch_size": 100, "update_cache": "yes"}

        hiter = cache.iteritems(hiter_config)

        start = time.time()
        count = 0
        while True:
            try:
                i = hiter.get_next()
            except StopIteration:
                print 'End of data, items read: ', count, ' with value ', i
                break
            count += 1
            if count % 100000 == 0:
                print count
        print "finshed after %d" % (time.time() - start)

        self.assertEqual(count, nparts)



    def test_get_row(self):
        from hfetch import connectCassandra
        from hfetch import Hcache

        keyspace = 'test'
        table = 'particle'
        num_keys = 10001

        config.session.execute("DROP TABLE IF EXISTS %s.%s;" %(keyspace, table))
        config.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid int, time float, ciao text,"
                               "x float, y float, z float, PRIMARY KEY(partid,time));" % (keyspace, table))

        for i in xrange(0, num_keys):
            vals = ','.join(str(e) for e in [i, i / .1, i / .2, i / .3, i / .4, "'" + str(i * 60) + "'"])
            config.session.execute(
                "INSERT INTO %s.%s(partid , time , x, y , z,ciao ) VALUES (%s)" % (keyspace, table, vals))

        contact_names = ['127.0.0.1']
        nodePort = 9042

        try:
            connectCassandra(contact_names, nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', contact_names, nodePort


        token_ranges = [(8070430489100699999, 8070450532247928832)]

        size = 10001

        keys = ["partid", "time"]
        values =  ["ciao", "x", "y", "z"]

        cache_config = {'cache_size': size}

        cache = Hcache(keyspace, table, "", token_ranges, keys,values,cache_config)


        # clustering key
        t1 = time.time()
        for pk in xrange(0, num_keys):
            ck = pk * 10
            try:
                result = cache.get_row([pk, ck])
                self.assertEqual(len(result), len(values))
            except KeyError as e:
                print "Error when retrieving value from cache:", e, [pk, ck]



        print 'time - load C++ cache with cassandra data: ', time.time() - t1

        t1 = time.time()
        for pk in xrange(0, num_keys):
            ck = pk * 10
            try:
                result = cache.get_row([pk, ck])
                self.assertEqual(len(result), len(values))
            except KeyError as e:
                print "Error when retrieving value from cache:", e, [pk, ck]
        # print 'items in res: ',len(result)
        print 'time - read data from C++ cache: ', time.time() - t1

        py_dict = {}
        cache = Hcache(keyspace, table, "", [(8070430489100699999, 8070450532247928832)], ["partid", "time"],
                       ["ciao", "x", "y", "z"], {'cache_size': num_keys})

        t1 = time.time()
        for pk in xrange(0, num_keys):
            ck = pk * 10
            try:
                result = cache.get_row([pk, ck])
                py_dict[(pk, ck)] = result
                self.assertEqual(len(result), len(values))
            except KeyError as e:
                print "Error when retrieving value from cache:", e, [pk, ck]
        print 'time - load data into python dict: ', time.time() - t1
        # print 'size ', len(py_dict)
        # print 'items in res: ',len(py_dict[1])

        t1 = time.time()
        for pk in xrange(0, num_keys):
            ck = pk * 10
            try:
                result = py_dict[(pk,ck)]
                self.assertEqual(len(result), len(values))
            except KeyError as e:
                print "Error when retrieving value from cache:", e, [pk, ck]
        print 'time - read data from the python dict: ', time.time() - t1
        # print 'size ', len(py_dict)
        # print 'items in res: ',len(py_dict[1])


    def write_text(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        '''''''''
        Simple test to store text and retrieve it
        '''''''''


        keyspace = "test"
        table = "bulk"

        config.session.execute("DROP TABLE IF EXISTS %s.%s;" %(keyspace, table))
        config.session.execute("CREATE TABLE %s.%s(partid int PRIMARY KEY, data text);" % (keyspace, table))


        contact_names = ['127.0.0.1']
        nodePort = 9042

        try:
            connectCassandra(contact_names, nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', contact_names, nodePort

        tokens=[]

        keys = ["partid"]
        values = ["data"]

        hcache_config = {'cache_size': '10', 'writer_buffer': 20}


        # CREATE TABLE test.bulk(partid int PRIMARY KEY, data text);
        cache = Hcache(keyspace, table, "",tokens, keys, values, hcache_config)

        for i in xrange(pow(10, 3)):
            cache.put_row([i], ['someRandomText'])


        hiter = cache.iteritems(10)

        #TODO something




    def test_iterators(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        '''''''''
        This test iterates over a huge amount of data, also update the cache
        '''''''''


        keyspace = "test"
        table = "words"
        num_keys = 20


        config.session.execute("DROP TABLE IF EXISTS %s.%s;" %(keyspace, table))
        config.session.execute(
            "CREATE TABLE %s.%s(position int PRIMARY KEY, wordinfo text);" % (keyspace, table))



        for i in xrange(0, num_keys):
            vals = ','.join(str(e) for e in [i,"'someRandomTextForTesting purposes - " + str(i * 60) + "'"])
            config.session.execute(
                "INSERT INTO %s.%s(position , wordinfo ) VALUES (%s)" % (keyspace, table, vals))


        contact_names = ['127.0.0.1']
        nodePort = 9042

        try:
            connectCassandra(contact_names, nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', contact_names, nodePort

        tkns = [(pow(-2, 63)+1, pow(2, 63)-1)]
        keys = ["position"]
        values = ["wordinfo"]
        hcache_config = {'cache_size': 100, 'writer_buffer': 20}

        cache = Hcache(keyspace,table, "WHERE token(position)>=? AND token(position)<?;", tkns,keys, values,hcache_config)


        iter_config = {"prefetch_size": 100, "update_cache": "yes"}
        myIter = cache.iteritems(iter_config)

        data = []
        for i in xrange(0, 10):
            data.append(myIter.get_next())

        assert (len(data) > 0)
        first_data = data[0]

        assert (len(first_data) == 2)
        first_key = [first_data[0]]

        assert (type(first_key[0]) == int)
        somedata = cache.get_row(first_key)
        #self.assertEqual((first_key + somedata), first_data)
        assert ((first_key + somedata) == first_data)

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



    def small_brute(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        '''''''''
        This test iterates over a small amount of data
        '''''''''


        keyspace = "test"
        table = "particle"
        nelems = 10001

        config.session.execute("DROP TABLE IF EXISTS %s.%s;" %(keyspace, table))
        config.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid int, time float, ciao text,"
                               "x float, y float, z float, PRIMARY KEY(partid,time));" % (keyspace, table))

        for i in xrange(0, nelems):
            vals = ','.join(str(e) for e in [i, i / .1, i / .2, i / .3, i / .4, "'" + str(i * 60) + "'"])
            config.session.execute(
                "INSERT INTO %s.%s(partid , time , x, y , z,ciao ) VALUES (%s)" % (keyspace, table, vals))



        contact_names = ['127.0.0.1']
        nodePort = 9042


        try:
            connectCassandra(contact_names, nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', contact_names, nodePort


        nblocks = 100

        t_f = pow(-2,63)  # Token begin range
        t_t = pow(2,63)
        # Token blocks
        tkn_size = (t_t - t_f) / (nelems / nblocks)
        tokens = [(a, a + tkn_size) for a in xrange(t_f, t_t - tkn_size, tkn_size)]

        hcache_config = {'cache_size': '10', 'writer_buffer': 20}
        keys = ["partid", "time"]
        values = ["time", "x"]

        cache = None
        # this should fail since a key can not be a column name at the same time (key=time, column=time)
        try:
            cache = Hcache(keyspace, table, "WHERE token(partid)>=? AND token(partid)<?;",
                       tokens, keys, values,hcache_config)
        except RuntimeError, e:
            self.assertTrue(True,e)


        keys = ["partid", "time"]
        values = ["x"]
        # now this should work
        try:
            cache = Hcache(keyspace, table, "WHERE token(partid)>=? AND token(partid)<?;",
                           tokens, keys, values, hcache_config)
        except RuntimeError, e:
            self.assertFalse(True,e)

        to = cache.iterkeys(10000)

        counter = 0
        while True:
            try:
                res = to.get_next()
                self.assertEqual(len(res),len(keys)+len(values))
            except StopIteration:
                break
            counter = counter + 1

        self.assertEqual(counter,nelems)



    def simpletest(self):
        from hfetch import connectCassandra
        from hfetch import Hcache

        keyspace = 'test'
        table = 'particle'
        nelems = 500

        config.session.execute("DROP TABLE IF EXISTS %s.%s;" %(keyspace, table))
        config.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid int, time float, ciao text,"
                               "x float, y float, z float, PRIMARY KEY(partid,time));" % (keyspace, table))

        for i in xrange(0, nelems):
            vals = ','.join(str(e) for e in [i, i / .1, i / .2, i / .3, i / .4, "'" + str(i * 60) + "'"])
            config.session.execute(
                "INSERT INTO %s.%s(partid , time , x, y , z,ciao ) VALUES (%s)" % (keyspace, table, vals))

        contact_names = ['127.0.0.1']
        nodePort = 9042

        try:
            connectCassandra(contact_names, nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', contact_names, nodePort

        keys = ["partid", "time"]
        values = ["x", "y", "z"]
        token_ranges = []
        # empty configuration parameter (the last dictionary) means to use the default config
        table = Hcache(keyspace,table,"WHERE token(partid)>=? AND token(partid)<?;", token_ranges,keys,values,{})

        def get_data(cache, keys):
            data = None
            try:
                data = cache.get_row(keys)
                self.assertEqual(len(data),len(values))
            except KeyError:
                print 'not found'
            return data

        q1 = get_data(table, [433, 4330])  # float(0.003)
        lost = get_data(table, [133, 1330])
        lost = get_data(table, [433, 4330])
        q2 = get_data(table, [433, 4330])
        self.assertEqual(q1,q2)




    def get_row_key_error(self):
        from hfetch import connectCassandra
        from hfetch import Hcache

        keyspace = 'test'
        table = 'particle'
        num_keys = 10001

        config.session.execute("DROP TABLE IF EXISTS %s.%s;" %(keyspace, table))
        config.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid int, time float, ciao text,"
                               "x float, y float, z float, PRIMARY KEY(partid,time));" % (keyspace, table))

        for i in xrange(0, num_keys):
            vals = ','.join(str(e) for e in [i, i / .1, i / .2, i / .3, i / .4, "'" + str(i * 60) + "'"])
            config.session.execute(
                "INSERT INTO %s.%s(partid , time , x, y , z,ciao ) VALUES (%s)" % (keyspace, table, vals))


        contact_names = ['127.0.0.1']
        nodePort = 9042

        token_ranges = [(8070430489100699999, 8070450532247928832)]

        non_existent_keys = 10

        cache_size = num_keys + non_existent_keys

        try:
            connectCassandra(contact_names, nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', contact_names, nodePort
        keys = ["partid", "time"]
        values = ["ciao", "x", "y", "z"]
        cache = Hcache(keyspace, table, "", token_ranges, keys, values,
                       {'cache_size': cache_size})

        # Access the cache, which is empty and queries cassandra to retrieve the data
        t1 = time.time()
        error_counter = 0
        for pk in xrange(0, num_keys + non_existent_keys):
            ck = pk * 10
            try:
                result = cache.get_row([pk, ck])
                self.assertEqual(len(result),len(values))
            except KeyError as e:
                error_counter = error_counter + 1

        print 'Retrieved {0} keys in {1} seconds. {2} keys weren\'t found, {3} keys weren\'t supposed to be found'.format(
            unicode(str(num_keys), 'utf-8'),
            unicode(str(time.time() - t1), 'utf-8'), unicode(str(error_counter), 'utf-8'),
            unicode(str(non_existent_keys), 'utf-8'))

        self.assertEqual(error_counter, non_existent_keys)

        # Access the cache, which has already all the data and will ask cassandra only if
        # the keys asked are not present
        t1 = time.time()
        error_counter = 0
        for pk in xrange(0, num_keys + non_existent_keys):
            ck = pk * 10
            try:
                result = cache.get_row([pk, ck])
                self.assertEqual(len(result),len(values))
            except KeyError as e:
                error_counter = error_counter + 1

        print 'Retrieved {0} keys in {1} seconds. {2} keys weren\'t found, {3} keys weren\'t supposed to be found'.format(
            unicode(str(num_keys), 'utf-8'),
            unicode(str(time.time() - t1), 'utf-8'), unicode(str(error_counter), 'utf-8'),
            unicode(str(non_existent_keys), 'utf-8'))

        self.assertEqual(error_counter, non_existent_keys)


    def uuid_test(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        import uuid
        '''''''''
        Simple test to store text and retrieves it
        '''''''''

        keyspace = "test"
        table = "uuid"

        config.session.execute("DROP TABLE IF EXISTS %s.%s;" %(keyspace, table))
        config.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid uuid, data int, PRIMARY KEY(partid));" % (keyspace, table))


        contact_names = ['127.0.0.1']
        nodePort = 9042


        nelem = 1000
        nblocks = 10

        t_f = pow(-2,63)  # Token begin range
        t_t = pow(2,63)
        # Token blocks
        tkn_size = (t_t - t_f) / (nelem / nblocks)
        tokens = [(a, a + tkn_size) for a in xrange(t_f, t_t - tkn_size, tkn_size)]



        try:
            connectCassandra(contact_names, nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', contact_names, nodePort

        keys = ["partid"]
        values = ["data"]

        # CREATE TABLE test.bulk(partid int PRIMARY KEY, data text);
        cache = Hcache(keyspace, table, "WHERE token(partid)>=? AND token(partid)<?;",
                       tokens, keys, values
                   , {'cache_size': '10', 'writer_buffer': 20})

        i = 0
        while i < pow(10, 3):
            u = uuid.uuid4()  # ('81da81e8-1914-11e7-908d-ecf4bb4c66c4')
            cache.put_row([u], [i])
            i += 1

        itera = cache.iteritems(10)
        try:
            L = uuid.UUID(itera.get_next()[0])
        except StopIteration:
            self.assertFalse()



    def words_test_hiter(self):
        from hfetch import connectCassandra
        from hfetch import HIterator
        '''
        This test iterates over huge lines of text
        '''



        contact_names = ['127.0.0.1']
        nodePort = 9042

        try:
            connectCassandra(contact_names, nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', contact_names, nodePort


        nelem = 10
        nblocks = 2

        t_f = pow(-2,63)  # Token begin range
        t_t = pow(2,63)
        # Token blocks
        tkn_size = (t_t - t_f) / (nelem / nblocks)
        tokens = [(a, a + tkn_size) for a in xrange(t_f, t_t - tkn_size, tkn_size)]


        itera = HIterator("hecuba_test", "wordsso_words",
                          tokens, ["position"], ["wordinfo"],
                          {'prefetch_size': '100', 'writer_buffer': 20})

        data = None
        while True:
            try:
                data = itera.get_next()
            except StopIteration:
                break

        print data


    def write_test(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        from hfetch import HWriter

        '''''''''
        This test iterates over a huge amount of data
        '''''''''
        keyspace = "test"
        table = "particle"
        table_write = "particle_write"
        nparts = 6000  # Num particles in range

        config.session.execute("DROP TABLE IF EXISTS %s.%s;" %(keyspace, table))
        config.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid int, time float, ciao text,"
                               "x float, y float, z float, PRIMARY KEY(partid,time));" % (keyspace, table))

        config.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid int, time float,"
                               "x float, y float, z float, PRIMARY KEY(partid,time));" % (keyspace, table_write))

        for i in xrange(0, nparts):
            vals = ','.join(str(e) for e in [i, i / .1, i / .2, i / .3, i / .4, "'" + str(i * 60) + "'"])
            config.session.execute(
                "INSERT INTO %s.%s(partid , time , x, y , z,ciao ) VALUES (%s)" % (keyspace, table, vals))

        contact_names = ['127.0.0.1']
        nodePort = 9042

        try:
            connectCassandra(contact_names, nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', contact_names, nodePort

        p = 1000  # Num partitions

        t_f = -7764607523034234880  # Token begin range
        # t_t = 5764607523034234880  # Token end range
        t_t = 7764607523034234880
        # Token blocks
        tkn_size = (t_t - t_f) / (nparts / p)
        tkns = [(a, a + tkn_size) for a in xrange(t_f, t_t - tkn_size, tkn_size)]

        a = Hcache("test", "particle", "WHERE token(partid)>=? AND token(partid)<?;", tkns, ["partid", "time"],
                   ["x", "y", "z"],
                   {'cache_size': '100', 'writer_buffer': 20})
        writer = HWriter("test", "particle_write", ["partid", "time"], ["x", "y", "z"],
                         {'writer_buffer': 20})

        def readAll(iter, wr):
            count = 1
            i = "random"
            while (i is not None):
                try:
                    i = iter.get_next()
                except StopIteration:
                    print 'End of data, items read: ', count, ' with value ', i
                    break
                wr.write(i[0:2], i[2:5])
                count += 1
                if count % 100000 == 0:
                    print count
            print "iter has %d elements" % count

        start = time.time()
        readAll(a.iteritems({"prefetch_size": 100, "update_cache": "yes"}), writer)
        print "finshed into %d" % (time.time() - start)
