#import gc
#gc.set_debug(gc.DEBUG_STATS|gc.DEBUG_LEAK)
from hfetch import *
import numpy as np
import time
from cassandra.cluster import *


def test_multidim():
    dims = 3
    elem_dim = 5

    print 'Running multidimensional partition test'
    print 'Dimensions: ', dims, ' Element in each dim: ', elem_dim

    session.execute("DROP TABLE if exists test.arrays;")
    session.execute("CREATE TABLE test.arrays(partid int PRIMARY KEY, image_block blob, image_block_pos int);")

    a = Hcache(keyspace, "arrays", "WHERE token(partid)>=? AND token(partid)<?;",
               [], ["partid"],
               [{"name": "image_block", "type": "int", "dims": "5x5x5", "partition": "true"}, "image_block_pos"], {})

    #prepare data
    bigarr = np.arange(pow(elem_dim, dims)).reshape(elem_dim, elem_dim, elem_dim)
    keys = [100]
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
    session.execute("DROP TABLE test.arrays;")


def nopart():

    elem_dim = 128
    txt_elem_dim = str(elem_dim)
    dims = 2

    print 'Running no partition test'
    print 'Dimensions: ', dims, ' Element in each dim: ', elem_dim

    session.execute("DROP TABLE if exists test.arrays;")
    session.execute("CREATE TABLE test.arrays(partid int PRIMARY KEY, image_block blob);")


    a = Hcache("test", "arrays", "WHERE token(partid)>=? AND token(partid)<?;",
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
    print '2D, elem dimension: ', elem_dim

    time.sleep(2)
    session.execute("DROP TABLE test.arrays;")



def part():

    dims = 2
    elem_dim = 2048
    txt_elem_dim = str(elem_dim)

    print 'Running partition test'
    print 'Dimensions: ', dims, ' Element in each dim: ', elem_dim

    session.execute("DROP TABLE if exists test.arrays;")
    session.execute("CREATE TABLE test.arrays(partid int PRIMARY KEY, image_block blob, image_block_pos int);")


    a = Hcache("test", "arrays", "WHERE token(partid)>=? AND token(partid)<?;",
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
    session.execute("DROP TABLE test.arrays;")



def npy_uuid():

    dims = 2
    elem_dim = 2048
    txt_elem_dim = str(elem_dim)

    print 'Running npy_uuid test'
    print 'Dimensions: ', dims, ' Element in each dim: ', elem_dim

    session.execute("DROP TABLE if exists test.arrays;")
    session.execute("CREATE TABLE test.arrays(partid int PRIMARY KEY, image uuid);")

    session.execute("CREATE TABLE test.arrays_aux(uuid uuid,  position int, data blob, PRIMARY KEY (uuid,position));")


    print 'create'
    a = Hcache("test", "arrays", "WHERE token(partid)>=? AND token(partid)<?;",
               [], ["partid"],[{"name": "image",
                                 "type": "double",
                                 "dims": txt_elem_dim + 'x' + txt_elem_dim,
                                 "partition": "true",
                                 "npy_table": "arrays_aux"}], {})

    print 'get'
    bigarr = np.arange(pow(elem_dim, 2)).reshape(elem_dim, elem_dim)
    bigarr.itemset(0, 14.0)
    print 'Array to be written', bigarr.astype('d')
    import time
    t1 = time.time()

    print 'put'
    keys = [300]
    a.put_row(keys, [bigarr.astype('d')])
    print 'Elapsed time', time.time() - t1
    print '2D, elem dimension: ', elem_dim

    time.sleep(2)
    session.execute("DROP TABLE test.arrays;")
    session.execute("DROP TABLE test.arrays_aux;")



def arr_put_get():
    dims = 2
    elem_dim = 2048
    txt_elem_dim = str(elem_dim)


    print 'Running npy_uuid test'
    print 'Dimensions: ', dims, ' Element in each dim: ', elem_dim


    session.execute("DROP TABLE if exists test.arrays;")
    session.execute("CREATE TABLE test.arrays(partid int PRIMARY KEY, image uuid);")

    session.execute("CREATE TABLE test.arrays_aux(uuid uuid,  position int, data blob, PRIMARY KEY (uuid,position));")


    a = Hcache("test", "arrays", "WHERE token(partid)>=? AND token(partid)<?;",
               [(-8070430489100700000, 8070450532247928832)], ["partid"], [
                   {"name": "image_block", "type": "double", "dims": txt_elem_dim + 'x' + txt_elem_dim,
                    "partition": "true"}, "image_block_pos"], {})

    bigarr = np.arange(pow(elem_dim, dims)).reshape(elem_dim, elem_dim)
    bigarr.itemset(0, 14.0)
    print 'Array to be written', bigarr.astype('d')
    import time
    t1 = time.time()
    keys = [300]
    a.put_row(keys, [bigarr.astype('d')])
    time.sleep(3)
    # othw we ask for the row before it has been processed
    result = a.get_row(keys)
    print 'Written:', bigarr.astype('d')
    resarr = result[0]
    print "And the result is... ", resarr.reshape((2048, 2048))
    print 'Elapsed time', time.time() - t1
    print '2D, elem dimension: ', elem_dim


    session.execute("DROP TABLE test.arrays;")
    session.execute("DROP TABLE test.arrays_aux;")



nodePort=9042
host_list=["127.0.0.1"]
keyspace = "test"

cluster = Cluster(host_list, nodePort)
session = cluster.connect(keyspace)




if __name__ == '__main__':

    connectCassandra(host_list,nodePort)

    test_multidim()

    nopart()

    part()

    npy_uuid()

    arr_put_get()

    wait = raw_input("End test?")
    cluster.shutdown()