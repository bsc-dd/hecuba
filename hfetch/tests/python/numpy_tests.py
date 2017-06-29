import unittest
import time

from cassandra.cluster import Cluster



class Hfetch_Tests(unittest.TestCase):
    keyspace = "hnumpy_test"
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

    def test_multidim(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        import numpy as np
        '''''''''
        Multidimensional partition test: A numpy of more than 2 dimensions is partitioned and stored at the same table
        
        Analyzes:
        
        '''''''''
        dims = 3
        elem_dim = 5

        try:
            connectCassandra(self.contact_names, self.nodePort)
        except RuntimeError, e:
            print e
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort


        table = "arrays"

        self.session.execute("DROP TABLE if exists test.arrays;")
        self.session.execute("CREATE TABLE test.arrays(partid int PRIMARY KEY, image_block blob, image_block_pos int);")

        a = Hcache(self.keyspace, table, "WHERE token(partid)>=? AND token(partid)<?;",
                   [], ["partid"],
                   [{"name": "image_block", "type": "int", "dims": "5x5x5", "partition": "true"},
                    "image_block_pos"], {})

        #prepare data
        bigarr = np.arange(pow(elem_dim, dims)).reshape(elem_dim, elem_dim, elem_dim)
        temp =100
        keys = [temp]
        values = [bigarr.astype('i')]

        #insert
        a.put_row(keys, values)

        # othw we ask for the row before it has been processed
        time.sleep(2)

        #retrieve
        keys = [100]
        result = a.get_row(keys)
        if np.array_equal(bigarr,result[0]):
            print 'Created and retrieved are equal'
        else:
            print 'Created and retrieved arrays differ, sth went wrong '
            print 'Array sent ', bigarr
            print 'Array retrieved ', result[0]
        time.sleep(2)
        self.session.execute("DROP TABLE test.arrays;")


    def test_nopart(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        import numpy as np
        '''''''''
        No partition test: An array is stored without slicing as raw bytes.
        
        Analyzes:
        
        '''''''''

        elem_dim = 128
        txt_elem_dim = str(elem_dim)
        dims = 2


        try:
            connectCassandra(self.contact_names, self.nodePort)
        except RuntimeError, e:
            print e
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort


        table = "arrays"

        self.session.execute("DROP TABLE if exists test.arrays;")
        self.session.execute("CREATE TABLE test.arrays(partid int PRIMARY KEY, image_block blob);")


        a = Hcache("test", table, "WHERE token(partid)>=? AND token(partid)<?;",
                   [], ["partid"],
                   [{"name": "image_block",
                     "type": "double",
                     "dims": txt_elem_dim + 'x' + txt_elem_dim,
                     "partition": "false"}], {})

        keys = [300]
        bigarr = np.arange(pow(elem_dim, dims)).reshape(elem_dim, elem_dim)
        bigarr.itemset(0, 14.0)
        print 'Array to be written', bigarr.astype('d')

        t1 = time.time()
        a.put_row(keys, [bigarr.astype('d')])
        print 'Elapsed time', time.time() - t1

        time.sleep(2)
        self.session.execute("DROP TABLE test.arrays;")



    def test_part(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        import numpy as np
        '''''''''
        Running partition test
        
        Analyzes:
        
        '''''''''

        dims = 2
        elem_dim = 2048
        txt_elem_dim = str(elem_dim)

        self.session.execute("DROP TABLE if exists test.arrays;")
        self.session.execute("CREATE TABLE test.arrays(partid int , image_block blob, image_block_pos int, PRIMARY KEY(partid,image_block_pos));")

        try:
            connectCassandra(self.contact_names, self.nodePort)
        except RuntimeError, e:
            print e
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort

        table = "arrays"

        a = Hcache("test", table, "WHERE token(partid)>=? AND token(partid)<?;",
                   [(-8070430489100700000, 8070450532247928832)], ["partid"],
                   [{"name": "image_block",
                     "type": "double",
                     "dims": txt_elem_dim + 'x' + txt_elem_dim,
                     "partition": "true"},
                    "image_block_pos"], {})

        keys = [300]
        bigarr = np.arange(pow(elem_dim, dims)).reshape(elem_dim, elem_dim)
        bigarr.itemset(0, 14.0)
        print 'Array to be written', bigarr.astype('d')

        t1 = time.time()
        a.put_row(keys, [bigarr.astype('d')])
        print 'Elapsed time', time.time() - t1
        print '2D, elem dimension: ', elem_dim

        time.sleep(2)
        self.session.execute("DROP TABLE test.arrays;")



    def test_npy_uuid(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        import numpy as np
        '''''''''
        Running npy_uuid test
        
        Analyzes:
        
        '''''''''

        dims = 2
        elem_dim = 2048
        txt_elem_dim = str(elem_dim)

        print 'Dimensions: ', dims, ' Element in each dim: ', elem_dim
        try:
            connectCassandra(self.contact_names, self.nodePort)
        except RuntimeError, e:
            print e
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort

        table = "arrays"

        self.session.execute("DROP TABLE if exists test.arrays;")
        self.session.execute("DROP TABLE if exists test.arrays_aux;")
        self.session.execute("CREATE TABLE test.arrays(partid int PRIMARY KEY, image uuid);")

        self.session.execute("CREATE TABLE test.arrays_aux(uuid uuid,  position int, data blob, PRIMARY KEY (uuid,position));")


        time.sleep(1)

        a = Hcache("test", table, "WHERE token(partid)>=? AND token(partid)<?;",
                   [], ["partid"],[{"name": "image",
                                     "type": "double",
                                     "dims": txt_elem_dim + 'x' + txt_elem_dim,
                                     "partition": "true",
                                     "npy_table": "arrays_aux"}], {})


        bigarr = np.arange(pow(elem_dim, 2)).reshape(elem_dim, elem_dim)
        bigarr.itemset(0, 14.0)
        #print 'Array to be written', bigarr.astype('d')

        t1 = time.time()
        #print a.get_row([300])
        a.put_row( [300],[bigarr.astype('d')])

        print 'Elapsed time', time.time() - t1
        print '2D, elem dimension: ', elem_dim

        time.sleep(3)
        self.session.execute("DROP TABLE test.arrays;")
        self.session.execute("DROP TABLE test.arrays_aux;")


    def test_arr_put_get(self):
        from hfetch import connectCassandra
        from hfetch import Hcache
        import numpy as np
        '''''''''
        Running arr_put_get test
        
        Analyzes:
        
        '''''''''
        dims = 2
        elem_dim = 2048
        txt_elem_dim = str(elem_dim)

        print 'Dimensions: ', dims, ' Element in each dim: ', elem_dim
        try:
            connectCassandra(self.contact_names, self.nodePort)
        except RuntimeError, e:
            print e
            print 'can\'t connect, verify the contact points and port', self.contact_names, self.nodePort

        table = "arrays"

        self.session.execute("DROP TABLE if exists test.arrays;")
        self.session.execute("CREATE TABLE test.arrays(partid int , image_block blob, image_block_pos int, PRIMARY KEY(partid,image_block_pos));")

        time.sleep(3)
        a = Hcache("test", table, "WHERE token(partid)>=? AND token(partid)<?;",
                   [], ["partid"], [
                       {"name": "image_block", "type": "double", "dims": txt_elem_dim + 'x' + txt_elem_dim,
                        "partition": "true"}, "image_block_pos"], {})

        bigarr = np.arange(pow(elem_dim, dims)).reshape(elem_dim, elem_dim)
        bigarr.itemset(0, 14.0)
        print 'Array to be written', bigarr.astype('d')

        t1 = time.time()
        keys =[300]
        a.put_row(keys, [bigarr.astype('d')])

        # othw we ask for the row before it has been processed
        time.sleep(5)

        try:
            result = a.get_row([300])
            resarr = result[0]
            print "And the result is... ", resarr.reshape((2048, 2048))
            print 'Elapsed time', time.time() - t1
            print '2D, elem dimension: ', elem_dim
        except KeyError:
            print 'not found'

        self.session.execute("DROP TABLE test.arrays;")