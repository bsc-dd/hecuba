import unittest
import time

from cassandra.cluster import Cluster



class Hfetch_Tests(unittest.TestCase):
    keyspace = "hfetch_test"
    contact_names = ['127.0.0.1']
    nodePort = 9042
    cluster = Cluster(contact_names,port=nodePort)
    session = cluster.connect()

    @classmethod
    def setUpClass(cls):
        cls.session.execute("CREATE KEYSPACE IF NOT EXISTS %s WITH replication "
                               "= {'class': 'SimpleStrategy', 'replication_factor': 1};" % cls.keyspace)

    @classmethod
    def tearDownClass(cls):
        #self.session.execute("DROP KEYSPACE IF EXISTS %s;" % cls.keyspace)
        pass

    def test_connection(self):
        from hfetch import connectCassandra

        # Test behaviour when NodePort is None (should return TypeError)
        test_contact_names = []
        test_node_port = None
        fails = False
        try:
            connectCassandra(test_contact_names,test_node_port)
        except TypeError:
            fails = True
        except Exception, e:
            self.fail(e.message)

        self.assertTrue(fails)
        fails = False

        # Test behaviour when contact_names is an empty text (should return ValueError)
        test_node_port = self.nodePort
        test_contact_names = [123456789]
        try:
            connectCassandra(test_contact_names,test_node_port)
        except TypeError:
            fails = True
        except Exception, e:
            self.fail(e.message)

        self.assertTrue(fails)
        fails = False

        # Test behaviour when contact_names is an empty text (should return ValueError)
        test_node_port = self.nodePort
        test_contact_names = ['']
        try:
            connectCassandra(test_contact_names,test_node_port)
        except ValueError:
            fails = True
        except Exception, e:
            self.fail(e.message)

        self.assertTrue(fails)
        fails = False

        #if no contact point specified, connects to 127.0.0.1
        try:
            self.contact_names.index('127.0.0.1')  #raises value error if not present
            test_contact_names = []
            connectCassandra(test_contact_names, test_node_port)
        except ValueError:
            pass
        except Exception, e:
            self.fail(e.message)

        test_node_port = self.nodePort
        test_contact_names = self.contact_names
        fails = False

        try:
            connectCassandra(test_contact_names, test_node_port)
        except RuntimeError:
            fails = True
        except Exception, e:
            self.fail(e.message)
        self.assertFalse(fails)


    def test_write_nulls_simple(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        '''''''''
        Simple test to store text and retrieve it

        Analyzes:
        - HCache
        - Put_row (write data mixed with nulls)
        '''''''''

        table = "nulls"

        self.session.execute("DROP TABLE IF EXISTS %s.%s;" % (self.keyspace, table))
        self.session.execute("CREATE TABLE %s.%s(partid int PRIMARY KEY, time float, data text);" % (self.keyspace, table))

        num_items = int(pow(10, 3))

        try:
            connectCassandra(self.contact_names, self.nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort

        nblocks = 10
        t_f = pow(-2, 63)  # Token begin range
        t_t = pow(2, 63) - 1
        # Token blocks
        tkn_size = (t_t - t_f) / (num_items / nblocks)
        tokens = [(a, a + tkn_size) for a in xrange(t_f, t_t - tkn_size, tkn_size)]

        keys = ["partid"]
        values = ["time","data"]

        hcache_config = {'cache_size': '10', 'writer_buffer': 20}

        cache = Hcache(self.keyspace, table, "", tokens, keys, values, hcache_config)
        import random
        for i in xrange(0, num_items):
            cache.put_row([i], [12,None])#random.sample({i,None},1)+random.sample({'SomeRandomText',None},1))
        time.sleep(10)

    def test_iterate_brute(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        '''''''''
        This test iterates over a huge amount of data and checks no data is lost
        
        Analyzes:
        - HCache
        - Iteritems from HCache
        - Updates the HCache with the prefetched data (iteritems)
        '''''''''

        table = "particle"
        nparts = 10000  # Num particles in range

        self.session.execute("DROP TABLE IF EXISTS %s.%s;" %(self.keyspace, table))
        self.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid int, time float, ciao text,"
                               "x float, y float, z float, PRIMARY KEY(partid,time));"% (self.keyspace,table))

        for i in xrange(0,nparts):
            vals = ','.join(str(e) for e in [i,i/.1,i/.2,i/.3,i/.4,"'"+str(i*60)+"'"])
            self.session.execute("INSERT INTO %s.%s(partid , time , x, y , z,ciao ) VALUES (%s)"% (self.keyspace,table,vals))

        try:
            connectCassandra(self.contact_names, self.nodePort)
        except RuntimeError, e:
            print e
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort

        p = 100  # Num partitions

        t_f = pow(-2,63)  # Token begin range
        t_t = pow(2,63)-1
        # Token blocks
        tkn_size = (t_t - t_f) / (nparts / p)
        tkns = [(a, a + tkn_size) for a in xrange(t_f, t_t - tkn_size, tkn_size)]

        keys = ["partid", "time"]
        values = ["x"]

        hcache_config = {'cache_size': '100', 'writer_buffer': 20}

        token_query = "WHERE token(partid)>=? AND token(partid)<?;"



        cache = Hcache(self.keyspace,table, token_query , tkns, keys, values, hcache_config)

        hiter_config = {"prefetch_size": 100, "update_cache": "yes"}

        hiter = cache.iteritems(hiter_config)

        count = 0
        start = time.time()
        while True:
            try:
                i = hiter.get_next()
                self.assertEqual(len(i),len(keys)+len(values))
            except StopIteration:
                break
            count += 1

        self.assertEqual(count, nparts)
        print "finshed after %d" % (time.time() - start)

    def test_type_error(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        '''''''''
        This test iterates over a set of particles, performing get_row operations

        Analyzes:
        - HCache
        - Get_row (setting TypeError properly)
        '''''''''

        self.keyspace = 'test'
        table = 'particle'
        num_keys = 10000 #num keys must be multiple of expected_errors
        expected_errors = 10

        self.session.execute("DROP TABLE IF EXISTS %s.%s;" % (self.keyspace, table))
        self.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid int, time float, ciao text,"
                               "x float, y float, z float, PRIMARY KEY(partid,time));" % (self.keyspace, table))

        for i in xrange(0, num_keys):
            vals = ','.join(str(e) for e in [i, i / .1, i / .2, i / .3, i / .4, "'" + str(i * 60) + "'"])
            self.session.execute(
                "INSERT INTO %s.%s(partid , time , x, y , z,ciao ) VALUES (%s)" % (self.keyspace, table, vals))

        try:
            connectCassandra(self.contact_names, self.nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort

        token_ranges = []

        cache_size = 10000

        keys = ["partid", "time"]
        values = ["ciao", "x", "y", "z"]

        cache_config = {'cache_size': cache_size}

        cache = Hcache(self.keyspace, table, "", token_ranges, keys, values, cache_config)

        type_errors = 0
        # clustering key

        for pk in xrange(0, num_keys):
            ck = pk * 10
            if pk % (num_keys/expected_errors) == 0:
                pk = 'wrong'
            try:
                result = cache.get_row([pk, ck])
                self.assertEqual(len(result), len(values))
            except KeyError as e:
                self.fail("Error when retrieving value from cache: "+str(e)+" -- "+str([pk, ck]))
            except TypeError as e:
                print "mykey ", ck/10
                print e.message
                type_errors = type_errors + 1

        self.assertEqual(expected_errors,type_errors)



    def test_get_row(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        '''''''''
        This test iterates over a set of particles, performing get_row operations
        
        Analyzes:
        - HCache (multiple reads of the same key)
        - Get_row
        '''''''''

        self.keyspace = 'test'
        table = 'particle'
        num_keys = 10001

        self.session.execute("DROP TABLE IF EXISTS %s.%s;" %(self.keyspace, table))
        self.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid int, time float, ciao text,"
                               "x float, y float, z float, PRIMARY KEY(partid,time));" % (self.keyspace, table))

        for i in xrange(0, num_keys):
            vals = ','.join(str(e) for e in [i, i / .1, i / .2, i / .3, i / .4, "'" + str(i * 60) + "'"])
            self.session.execute(
                "INSERT INTO %s.%s(partid , time , x, y , z,ciao ) VALUES (%s)" % (self.keyspace, table, vals))


        try:
            connectCassandra(self.contact_names, self.nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort


        token_ranges = []

        cache_size = 10001

        keys = ["partid", "time"]
        values =  ["ciao", "x", "y", "z"]

        cache_config = {'cache_size': cache_size}

        cache = Hcache(self.keyspace, table, "", token_ranges, keys,values,cache_config)


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
        cache = Hcache(self.keyspace, table, "", [(8070430489100699999, 8070450532247928832)], ["partid", "time"],
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


    def test_put_row_text(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        '''''''''
        Simple test to store text and retrieve it
        
        Analyzes:
        - HCache
        - Put_row (write text)
        - Iteritems (read text)
        '''''''''

        table = "bulk"

        self.session.execute("DROP TABLE IF EXISTS %s.%s;" %(self.keyspace, table))
        self.session.execute("CREATE TABLE %s.%s(partid int PRIMARY KEY, data text);" % (self.keyspace, table))


        num_items = int(pow(10, 3))

        try:
            connectCassandra(self.contact_names, self.nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort

        nblocks=10
        t_f = pow(-2, 63)  # Token begin range
        t_t = pow(2, 63) - 1
        # Token blocks
        tkn_size = (t_t - t_f) / (num_items / nblocks)
        tokens = [(a, a + tkn_size) for a in xrange(t_f, t_t - tkn_size, tkn_size)]


        keys = ["partid"]
        values = ["data"]

        hcache_config = {'cache_size': '10', 'writer_buffer': 20}


        cache = Hcache(self.keyspace, table, "",tokens, keys, values, hcache_config)
        for i in xrange(0,num_items):
            cache.put_row([i], ['someRandomText'])

        #it doesnt make sense to count the read elements
        # because the data is still being written async
        hiter = cache.iteritems(10)
        while True:
            try:
                data = hiter.get_next()
                self.assertEqual(len(data),len(keys)+len(values))
                self.assertEqual(data[1],'someRandomText')
            except StopIteration:
                break


    def test_iterators(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        '''''''''
        This test iterates over some text and check coherency between hcache and hiter
        
        Analyzes:
        - HCache
        - Get_row (read text)
        - Iteritems (read text)
        '''''''''


        
        table = "words"
        num_keys = 20


        self.session.execute("DROP TABLE IF EXISTS %s.%s;" %(self.keyspace, table))
        self.session.execute(
            "CREATE TABLE %s.%s(position int PRIMARY KEY, wordinfo text);" % (self.keyspace, table))



        for i in xrange(0, num_keys):
            vals = ','.join(str(e) for e in [i,"'someRandomTextForTesting purposes - " + str(i * 60) + "'"])
            self.session.execute(
                "INSERT INTO %s.%s(position , wordinfo ) VALUES (%s)" % (self.keyspace, table, vals))

        try:
            connectCassandra(self.contact_names, self.nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort

        tkns = [(pow(-2, 63)+1, pow(2, 63)-1)]
        keys = ["position"]
        values = ["wordinfo"]
        hcache_config = {'cache_size': 100, 'writer_buffer': 20}

        cache = Hcache(self.keyspace,table, "WHERE token(position)>=? AND token(position)<?;", tkns,keys, values,hcache_config)


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

        while True:
            try:
                i = myIter.get_next()
            except StopIteration:
                print 'End of data, items read: ', count, ' with value ', i
                break
            count = count + 1

        print 'data was: \n', data



    def test_small_brute(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        '''''''''
        This test iterates over a small amount of data using an iterkeys and validates that
        no column name can be a key and value at the same time
        
        Analyzes:
        - HCache (enforce column can't be key and value at the same time)
        - Iterkeys
        '''''''''


        
        table = "particle"
        nelems = 10001

        self.session.execute("DROP TABLE IF EXISTS %s.%s;" %(self.keyspace, table))
        self.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid int, time float, ciao text,"
                               "x float, y float, z float, PRIMARY KEY(partid,time));" % (self.keyspace, table))

        for i in xrange(0, nelems):
            vals = ','.join(str(e) for e in [i, i / .1, i / .2, i / .3, i / .4, "'" + str(i * 60) + "'"])
            self.session.execute(
                "INSERT INTO %s.%s(partid , time , x, y , z,ciao ) VALUES (%s)" % (self.keyspace, table, vals))

        try:
            connectCassandra(self.contact_names, self.nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort


        nblocks = 100

        t_f = pow(-2,63)  # Token begin range
        t_t = pow(2,63)-1
        # Token blocks
        tkn_size = (t_t - t_f) / (nelems / nblocks)
        tokens = [(a, a + tkn_size) for a in xrange(t_f, t_t - tkn_size, tkn_size)]

        hcache_config = {'cache_size': '10', 'writer_buffer': 20}
        keys = ["partid", "time"]
        values = ["time", "x"]

        cache = None
        # this should fail since a key can not be a column name at the same time (key=time, column=time)
        try:
            cache = Hcache(self.keyspace, table, "WHERE token(partid)>=? AND token(partid)<?;",
                       tokens, keys, values,hcache_config)
        except RuntimeError, e:
            self.assertTrue(True,e)


        keys = ["partid", "time"]
        values = ["x"]
        # now this should work
        try:
            cache = Hcache(self.keyspace, table, "WHERE token(partid)>=? AND token(partid)<?;",
                           tokens, keys, values, hcache_config)
        except RuntimeError, e:
            self.assertFalse(True,e)

        to = cache.iterkeys(10000)

        counter = 0
        while True:
            try:
                res = to.get_next()
                self.assertEqual(len(res),len(keys))
            except StopIteration:
                break
            counter = counter + 1

        self.assertEqual(counter,nelems)



    def test_simpletest(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        '''''''''
        
        Analyzes:
        '''''''''

        self.keyspace = 'test'
        table = 'particle'
        nelems = 500

        self.session.execute("DROP TABLE IF EXISTS %s.%s;" %(self.keyspace, table))
        self.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid int, time float, ciao text,"
                               "x float, y float, z float, PRIMARY KEY(partid,time));" % (self.keyspace, table))

        for i in xrange(0, nelems):
            vals = ','.join(str(e) for e in [i, i / .1, i / .2, i / .3, i / .4, "'" + str(i * 60) + "'"])
            self.session.execute(
                "INSERT INTO %s.%s(partid , time , x, y , z,ciao ) VALUES (%s)" % (self.keyspace, table, vals))

        try:
            connectCassandra(self.contact_names, self.nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort

        keys = ["partid", "time"]
        values = ["x", "y", "z"]
        token_ranges = []
        # empty configuration parameter (the last dictionary) means to use the default config
        table = Hcache(self.keyspace,table,"WHERE token(partid)>=? AND token(partid)<?;", token_ranges,keys,values,{})

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




    def test_get_row_key_error(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        '''''''''
        This test check the hcache sets a key error when the key we asked doesnt exist
        Analyzes:
        - Hcache
        - Get_row (returning KeyError)
        '''''''''

        self.keyspace = 'test'
        table = 'particle'
        num_keys = 10001

        self.session.execute("DROP TABLE IF EXISTS %s.%s;" %(self.keyspace, table))
        self.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid int, time float, ciao text,"
                               "x float, y float, z float, PRIMARY KEY(partid,time));" % (self.keyspace, table))

        for i in xrange(0, num_keys):
            vals = ','.join(str(e) for e in [i, i / .1, i / .2, i / .3, i / .4, "'" + str(i * 60) + "'"])
            self.session.execute(
                "INSERT INTO %s.%s(partid , time , x, y , z,ciao ) VALUES (%s)" % (self.keyspace, table, vals))


        token_ranges = [(8070430489100699999, 8070450532247928832)]

        non_existent_keys = 10

        cache_size = num_keys + non_existent_keys

        try:
            connectCassandra(self.contact_names, self.nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort
        keys = ["partid", "time"]
        values = ["ciao", "x", "y", "z"]
        cache = Hcache(self.keyspace, table, "", token_ranges, keys, values,
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
        This test check the correct handling of UUIDs
        
        Analyzes:
        - Hcache
        - Put_row
        - Iteritems
        '''''''''


        
        table = "uuid"

        self.session.execute("DROP TABLE IF EXISTS %s.%s;" %(self.keyspace, table))
        self.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid uuid, data int, PRIMARY KEY(partid));" % (self.keyspace, table))

        nelem = 1000
        nblocks = 10

        t_f = pow(-2,63)  # Token begin range
        t_t = pow(2,63)-1
        # Token blocks
        tkn_size = (t_t - t_f) / (nelem / nblocks)
        tokens = [(a, a + tkn_size) for a in xrange(t_f, t_t - tkn_size, tkn_size)]



        try:
            connectCassandra(self.contact_names, self.nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort

        keys = ["partid"]
        values = ["data"]

        # CREATE TABLE test.bulk(partid int PRIMARY KEY, data text);
        cache = Hcache(self.keyspace, table, "WHERE token(partid)>=? AND token(partid)<?;",
                       tokens, keys, values
                   , {'cache_size': '10', 'writer_buffer': 20})


        #Write data
        someid = None
        i = 0
        while i < nelem:
            u = uuid.uuid4()  # ('81da81e8-1914-11e7-908d-ecf4bb4c66c4')
            cache.put_row([u], [i])
            if i==nelem/2:
                someid = u
            i += 1

        #by recreating the cache we wait until all the data is written

        cache = Hcache(self.keyspace, table, "WHERE token(partid)>=? AND token(partid)<?;",
                       tokens, keys, values
                       , {'cache_size': '10', 'writer_buffer': 20})
        #Read data
        itera = cache.iteritems(10)
        found = False
        counter = 0
        while True:
            try:
                L = uuid.UUID(itera.get_next()[0])
                if L == someid:
                    found = True
            except StopIteration:
                break
            counter = counter + 1

        self.assertEqual(counter, nelem)
        self.assertTrue(found)



    def words_test_hiter(self):
        from hfetch import connectCassandra
        from hfetch import HIterator
        import random
        import string
        '''
        This test iterates over huge lines of text and verifies the correct behaviour of HIterator
        By default it acts as an iteritems
        
        Analyzes:
        - HIterator
        - Iteritems
        '''


        table = "words"
        nelems=2000
        length_row=100


        self.session.execute("DROP TABLE IF EXISTS %s.%s;" % (self.keyspace, table))
        self.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(position int, wordinfo text, PRIMARY KEY(position));" % (self.keyspace, table))

        for i in xrange(0, nelems):
            vals = ','.join([str(i),"'"+''.join(random.choice(string.ascii_uppercase+string.ascii_lowercase+" "+ string.digits) for _ in range(length_row))+"'"])
            self.session.execute(
                "INSERT INTO %s.%s(position,wordinfo) VALUES (%s)" % (self.keyspace, table, vals))

        try:
            connectCassandra(self.contact_names, self.nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort


        nelem = 10
        nblocks = 2

        t_f = pow(-2,63)  # Token begin range
        t_t = pow(2,63)-1
        # Token blocks
        tkn_size = (t_t - t_f) / (nelem / nblocks)
        tokens = [(a, a + tkn_size) for a in xrange(t_f, t_t - tkn_size, tkn_size)]

        keys = ["position"]
        values = ["wordinfo"]
        hiter_config = {'prefetch_size': '100', 'writer_buffer': 20}

        itera = HIterator(self.keyspace, table, tokens, keys, values, hiter_config)

        while True:
            try:
                data = itera.get_next()
            except StopIteration:
                break



    def write_test(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        from hfetch import HWriter

        '''''''''
        While the iterator retrieves the data from a table, the writer stores it into another table
        
        Analyzes:
        - HCache
        - HWriter
        - Iteritems (updating the cache)
        '''''''''
        
        table = "particle"
        table_write = "particle_write"
        nparts = 6000  # Num particles in range

        self.session.execute("DROP TABLE IF EXISTS %s.%s;" %(self.keyspace, table))
        self.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid int, time float, ciao text,"
                               "x float, y float, z float, PRIMARY KEY(partid,time));" % (self.keyspace, table))

        self.session.execute("CREATE TABLE IF NOT EXISTS %s.%s(partid int, time float,"
                               "x float, y float, z float, PRIMARY KEY(partid,time));" % (self.keyspace, table_write))

        for i in xrange(0, nparts):
            vals = ','.join(str(e) for e in [i, i / .1, i / .2, i / .3, i / .4, "'" + str(i * 60) + "'"])
            self.session.execute(
                "INSERT INTO %s.%s(partid , time , x, y , z,ciao ) VALUES (%s)" % (self.keyspace, table, vals))


        try:
            connectCassandra(self.contact_names, self.nodePort)
        except Exception:
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort

        p = 1000  # Num partitions

        t_f = -7764607523034234880  # Token begin range
        # t_t = 5764607523034234880  # Token end range
        t_t = 7764607523034234880
        # Token blocks
        tkn_size = (t_t - t_f) / (nparts / p)
        tkns = [(a, a + tkn_size) for a in xrange(t_f, t_t - tkn_size, tkn_size)]
        keys = ["partid", "time"]
        values = ["x", "y", "z"]
        a = Hcache(self.keyspace, table, "WHERE token(partid)>=? AND token(partid)<?;", tkns, keys,
                   values,{self.keyspace: '100', 'writer_buffer': 20})

        writer = HWriter("test", table_write, keys, values,{'writer_buffer': 20})

        def readAll(iter, wr):
            count = 1
            while True:
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
