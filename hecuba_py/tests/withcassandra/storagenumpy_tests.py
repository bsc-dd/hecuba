import gc
import unittest

from hecuba import config, StorageNumpy
import uuid
import numpy as np

from storage.api import getByID

from time import time as timer
import random

class StorageNumpyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = config.execution_name
        config.execution_name = "StorageNumpyTest".lower()

    @classmethod
    def tearDownClass(cls):
        #config.session.execute("DROP KEYSPACE IF EXISTS {}".format(config.execution_name), timeout=60)
        config.execution_name = cls.old

    # Create a new keyspace per test
    def setUp(self):
        self.ksp = config.execution_name
        pass

    def tearDown(self):
        pass

    table = 'numpy_test'

    def test_init_empty(self):
        tablename = None

        base_array = np.arange(4096).reshape((64, 64))
        storage_id = None

        basic_init = StorageNumpy(base_array)
        self.assertTrue(np.array_equal(basic_init, base_array))

        complete_init = StorageNumpy(base_array, storage_id, tablename)
        self.assertTrue(np.array_equal(complete_init, base_array))

    def test_types_in_memory(self):
        base_array = np.arange(256)

        for typecode in np.typecodes['Integer']:
            typed_array = StorageNumpy(base_array.astype(typecode))
            self.assertTrue(np.array_equal(typed_array, base_array.astype(typecode)))

        for typecode in np.typecodes['UnsignedInteger']:
            typed_array = StorageNumpy(base_array.astype(typecode))
            self.assertTrue(np.array_equal(typed_array, base_array.astype(typecode)))

    def test_reconstruct(self):
        base_array = np.arange(256)
        tablename = self.ksp + '.' + "test_reconstruct"
        typecode = 'mytype'
        niter = 2

        for _ in range(niter):
            # Build array and store
            typed_array = StorageNumpy(base_array, tablename)
            self.assertTrue(np.array_equal(typed_array, base_array))
            typed_array.sync() # Flush values to cassandra
            # Load array
            typed_array = StorageNumpy(None, tablename)
            self.assertTrue(np.allclose(typed_array, base_array))
            typed_array.delete_persistent()

    def test_types_persistence(self):
        base_array = np.arange(256)
        tablename = self.ksp + '.' + "test_types_persistence"

        for typecode in np.typecodes['Integer']:
            if typecode == 'p':
                # TODO For now skip arrays made of pointers
                pass
            typed_array = StorageNumpy(base_array.astype(typecode), tablename)
            self.assertTrue(np.array_equal(typed_array, base_array.astype(typecode)))

            typed_array.sync() # Flush values to cassandra

            typed_array = StorageNumpy(None, tablename)
            self.assertTrue(np.allclose(typed_array, base_array.astype(typecode)))
            typed_array.delete_persistent()

        for typecode in np.typecodes['UnsignedInteger']:
            if typecode == 'P':
                # TODO For now skip arrays made of pointers
                pass
            typed_array = StorageNumpy(base_array.astype(typecode), tablename)
            self.assertTrue(np.allclose(typed_array, base_array.astype(typecode)))

            typed_array.sync() # Flush values to cassandra

            typed_array = StorageNumpy(None, tablename)
            self.assertTrue(np.allclose(typed_array, base_array.astype(typecode)))
            typed_array.delete_persistent()

    def test_read_all(self):
        nelem = 2 ** 21
        elem_dim = 2 ** 7

        base_array = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(input_array=base_array, name="test_read_all")

        casted.sync() # Flush values to cassandra
        test_numpy = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(name="test_read_all")
        chunk = casted[slice(None, None, None)]
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))
        casted.delete_persistent()

    def test_numpy_reserved_5d_read_all(self):

        nelem = 100000
        elem_dim = 10

        base_array = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(input_array=base_array, name="test_5d_read_all")

        casted.sync() # Flush values to cassandra

        test_numpy = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(name="test_5d_read_all")
        chunk = casted[slice(None, None, None)]
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))
        casted.delete_persistent()

    def test_explicit_construct(self):
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)

        basic_init = StorageNumpy()

    def test_view_cast(self):
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)

        base_array = np.arange(4096).reshape((64, 64))
        view_cast = base_array.view(StorageNumpy)

    def test_new_from_template(self):
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        base_array = np.arange(4096).reshape((64, 64))
        basic_init = StorageNumpy(base_array)
        new_from_template = basic_init[:32]

    def test_new2_from_template(self):
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        base_array = np.arange(4096).reshape((64, 64))
        basic_init = StorageNumpy(base_array)
        new_from_template = basic_init[32:]

    def test_get_subarray(self):
        base = np.arange(8 * 8 * 4).reshape((8, 8, 4))
        hecu_p = StorageNumpy(input_array=base, name='test_get_subarray')
        hecu_p.sync() # Flush values to cassandra
        hecu_r2 = StorageNumpy(name="test_get_subarray")
        res = hecu_r2[:3, :2]
        sum = res.sum()
        res = hecu_r2[:3, :2]
        avg = res.mean()
        self.assertGreater(sum, 0)
        self.assertGreater(avg, 0)

    def test_slicing_3d(self):
        base = np.arange(8 * 8 * 4).reshape((8, 8, 4))
        hecu = StorageNumpy(input_array=base, name='test_slicing_3d')
        res_hecu = hecu[6:7, 4:]
        res = base[6:7, 4:]
        self.assertTrue(np.array_equal(res, res_hecu))

        hecu.sync() # Flush values to cassandra
        hecu = StorageNumpy(name="test_slicing_3d")
        res_hecu = hecu[6:7, 4:]
        self.assertTrue(np.array_equal(res, res_hecu))

        hecu.delete_persistent()

    def test_slicing_ndims(self):
        import random
        ndims = 10
        max_elements = 2048
        for dims in range(1, ndims):
            elem_per_dim = int(max_elements ** (1 / dims))
            select = (slice(random.randint(0, elem_per_dim)),) * dims
            base = np.arange(elem_per_dim ** dims).reshape((elem_per_dim,) * dims)

            hecu = StorageNumpy(input_array=base, name='test_slicing_ndims')
            res_hecu = hecu[select]
            res = base[select]
            self.assertTrue(np.array_equal(res, res_hecu))

            hecu.sync() # Flush values to cassandra

            hecu = StorageNumpy(name="test_slicing_ndims")
            res_hecu = hecu[select]
            res = base[select]
            self.assertTrue(np.array_equal(res, res_hecu))
            hecu.delete_persistent()
            del res_hecu
            del hecu

    def test_slice_ops(self):
        obj = np.arange(8 * 8 * 8).reshape((8, 8, 8))
        hecu = StorageNumpy(input_array=obj, name='test_slice_ops')
        hecu_sub = hecu[:2, 3:, 4:]
        sum = hecu_sub.sum()
        self.assertGreater(sum, 0)
        description = repr(hecu_sub)
        self.assertIsInstance(description, str)
        hecu.delete_persistent()

    def test_slice_ops2(self):
        obj = np.arange(8 * 8 * 8).reshape((8, 8, 8))
        hecu = StorageNumpy(input_array=obj, name='test_slice_ops2')
        hecu_sub = hecu[:2, 3:, 4:]
        hecu_sub2 = hecu_sub[:1, 2:, 3:]
        sum = hecu_sub2.sum()
        self.assertGreater(sum, 0)
        description = repr(hecu_sub2)
        self.assertIsInstance(description, str)
        hecu.delete_persistent()

    def test_slice_from_numpy_array(self):
        obj = np.arange(8 * 8 * 8).reshape((8, 8, 8))
        hecu = StorageNumpy(input_array=obj, name='test_slice_numpy')
        l = np.array((0,1))
        hecu_sub = hecu[l]  #Access using an array of indexes
# FIXME add more testing, currently if it does not segfault, then it works
        sum = hecu_sub.sum()
        self.assertEqual(sum, obj[l].sum())
        hecu.delete_persistent()

    def test_iter_numpy(self):
        obj = np.arange(8 * 8 * 8).reshape((8, 8, 8))
        hecu = StorageNumpy(input_array=obj, name='test_iter_numpy')
        acc = 0
        for i in hecu:
            acc = acc + 1

        hecu_sub = hecu[:2, 3:, 4:]

        acc2 = 0
        for i in hecu_sub:
            acc2 = acc2 + 1

        self.assertGreater(acc, acc2)
        hecu.delete_persistent()

    def test_assign_slice(self):
        base = np.arange(8 * 8 * 4).reshape((8, 8, 4))
        hecu_p = StorageNumpy(input_array=base, name='test_assign_slice')
        sub_hecu = hecu_p[:2, 3:]
        sub_hecu[0][2:] = 0
        hecu_p.sync() # Flush values to cassandra
        hecu_p_load = StorageNumpy(name="test_assign_slice")
        rep = repr(hecu_p_load)
        self.assertIsInstance(rep, str)
        # StorageNumpy in memory and in database should share data
        load_sub_arr = hecu_p_load[:]
        self.assertFalse(np.array_equal(load_sub_arr, np.arange(8 * 8 * 4).reshape((8, 8, 4))))
        self.assertTrue(np.array_equal(sub_hecu, hecu_p_load[:2, 3:]))
        # Clean up
        hecu_p_load.delete_persistent()

    def test_assign_element(self):
        base = np.arange(8 * 8 * 4).reshape((8, 8, 4))
        hecu_p = StorageNumpy(input_array=base, name='test_assign_element')
        sub_hecu = hecu_p[:2, 3:]
        sub_hecu[0][1][0] = 0
        hecu_p.sync()
        hecu_p_load = StorageNumpy(name="test_assign_element")
        rep = repr(hecu_p_load)
        self.assertIsInstance(rep, str)
        load_sub_arr = hecu_p_load[:]
        self.assertFalse(np.array_equal(load_sub_arr, np.arange(8 * 8 * 4).reshape((8, 8, 4))))
        sub_hecu_load = hecu_p_load[:2, 3:]
        self.assertTrue(sub_hecu_load[0][1][0] == 0)
        # Clean up
        hecu_p_load.delete_persistent()

    def test_load_2_dif_clusters_same_instance(self):
        base = np.arange(50 * 50).reshape((50, 50))
        hecu_p = StorageNumpy(input_array=base, name='load_2_clustrs_same_inst')
        hecu_p.sync() # Flush values to cassandra
        hecu_p_load = StorageNumpy(name="load_2_clustrs_same_inst")
        hecu_p_load[0:1, 0:1]
        self.assertTrue(np.array_equal(hecu_p_load[40:50, 40:50], base[40:50, 40:50]))

    def test_split_by_rows(self):
        """
        Tests iterating through the rows of the Hecuba array
        """
        bn, bm = (1, 10)
        x = np.arange(100).reshape(10, -1)
        blocks = []
        for i in range(0, x.shape[0], bn):
            row = [x[i: i + bn, j: j + bm] for j in range(0, x.shape[1], bm)]
            blocks.append(row)

        data = StorageNumpy(input_array=x, name="test_split_by_rows")

        data.sync() # Flush values to cassandra
        for i, chunk in enumerate(data.np_split(block_size=(bn, bm))):
            storage_id = chunk.storage_id
            chunk.sync() #Flush data
            del chunk
            chunk = getByID(storage_id)
            self.assertTrue(np.array_equal(list(chunk), blocks[i]))

        self.assertEqual(i + 1, len(blocks))

    def test_split_by_columns(self):
        """
        Tests iterating through the columns of the Hecuba array
        """
        bn, bm = (10, 1)
        x = np.arange(100).reshape(10, -1)
        blocks = []
        for i in range(0, x.shape[0], bn):
            row = [x[i: i + bn, j: j + bm] for j in range(0, x.shape[1], bm)]
            blocks.append(row)

        data = StorageNumpy(input_array=x, name="test_split_by_columns")
        data.sync() # Flush values to cassandra
        for i, chunk in enumerate(data.np_split(block_size=(bn, bm))):
            storage_id = chunk.storage_id
            chunk.sync() #Flush data
            del chunk
            chunk = getByID(storage_id)
            self.assertTrue(np.array_equal(list(chunk), blocks[i]))

        self.assertEqual(i + 1, len(blocks))

    def test_split_rows_and_columns(self):

        bn, bm = (2, 1)
        x = np.arange(100).reshape(10, -1)
        blocks = []
        for i in range(0, x.shape[0], bn):
            row = [x[i: i + bn, j: j + bm] for j in range(0, x.shape[1], bm)]
            blocks.append(row)

        data = StorageNumpy(input_array=x, name="test_split_rows_and_columns")

        data.sync() # Flush values to cassandra
        for i, chunk in enumerate(data.np_split(block_size=(bn, bm))):
            storage_id = chunk.storage_id
            chunk.sync() #Flush data
            del chunk
            chunk = getByID(storage_id)
            self.assertTrue(np.array_equal(list(chunk), blocks[i]))

        self.assertEqual(i + 1, len(blocks))

    @unittest.skip("np_split is not maintained...")
    def test_split_already_persistent(self):

        bn, bm = (2, 1)
        x = np.arange(100).reshape(10, -1)
        blocks = []
        for i in range(0, x.shape[0], bn):
            row = [x[i: i + bn, j: j + bm] for j in range(0, x.shape[1], bm)]
            blocks.append(row)

        data = StorageNumpy(input_array=x, name="test_split_already_persistent")

        data.sync() # Flush values to cassandra
        for i, chunk in enumerate(data.np_split(block_size=(bn, bm))):
            storage_id = chunk.storage_id
            chunk.sync() #Flush data
            del chunk
            chunk = getByID(storage_id)
            self.assertTrue(np.array_equal(list(chunk), blocks[i]))

        del data
        gc.collect()

        data = StorageNumpy(name="test_split_already_persistent")
        self.assertTrue(np.array_equal(list(data), x))

        for i, chunk in enumerate(data.np_split(block_size=(bn, bm))):
            storage_id = chunk.storage_id
            chunk.sync() #Flush data
            del chunk
            chunk = getByID(storage_id)
            self.assertTrue(np.array_equal(list(chunk), blocks[i]))

        self.assertEqual(i + 1, len(blocks))

    def test_storagenumpy_copy_memory(self):
        #'''
        #Check that the memory from a StorageNumpy does not share original array
        #'''
        n = np.arange(12).reshape(3,4)

        s1 = StorageNumpy(n, "test_storagenumpy_copy_memory")

        # StorageNumpy s1 and n should NOT share memory
        s1[0][0] = 42
        self.assertTrue(not np.array_equal(s1, n))
        s1[0][0] = n[0][0] # Undo

        n[2][2] = 666
        self.assertTrue(not np.array_equal(s1, n))
        # Clean up
        s1.delete_persistent()


    def test_storagenumpy_from_storagenumpy(self):
        #'''
        #Create a StorageNumpy from another StorageNumpy
        #'''

        n = np.arange(12).reshape(3,4)

        s1 = StorageNumpy(n, "test_sn_from_sn")

        s2 = StorageNumpy(s1) # Create a StorageNumpy from another StorageNumpy

        self.assertTrue(s2.storage_id is None)
        self.assertTrue(s2._get_name() is None)
        self.assertTrue(np.array_equal(s2, n))

        # StorageNumpy s1 and s2 should not share memory
        s1[0][0] = 42
        self.assertTrue(s2[0,0] != s1[0,0])

        s2[2][2] = 666
        self.assertTrue(s2[2,2] != s1[2,2])

        # Create a third StorageNumpy
        s3 = StorageNumpy(s2)

        self.assertTrue(s3.storage_id is None)
        self.assertTrue(s3._get_name() is None)
        self.assertTrue(np.array_equal(s3, s2))

        # Clean up
        s1.delete_persistent()

    def test_storagenumpy_reshape(self):
        #'''
        #Reshape a StorageNumpy
        #'''

        n = np.arange(12).reshape(3,4)

        s1 = StorageNumpy(n, "test_storagenumpy_reshape")

        try:
            r = s1.view()
            r.shape = (4,3)
            self.assertTrue(r.storage_id == s1.storage_id)
        except: # If this exception code is executed means that a COPY is needed and therefore the resulting object is VOLATILE!
            r = s1.reshape(4,3)
            self.assertTrue(r.storage_id is None)
            self.assertTrue(r._is_persistent == False)
        self.assertTrue(r.shape != s1.shape)
        self.assertTrue(r.strides != s1.strides)


        # Clean up
        s1.delete_persistent()

    def test_transpose(self):
        #'''
        #Test the transpose
        #'''
        n=np.arange(12).reshape(3,4)

        s=StorageNumpy(n,"testTranspose")

        t=s.transpose()
        self.assertTrue(t[0,1] == s [1,0])

        t[0,1]=42

        self.assertTrue(t[0,1] == s[1,0])

        # Clean up
        s.delete_persistent()

    def test_copy_storageNumpyPersist(self):
        #'''
        #Test that a copy of a StorageNumpy does not share memory (Persistent version)
        #'''
        n=np.arange(12).reshape(3,4)

        s=StorageNumpy(n,"test_copy_storageNumpyPersist")
        c=s.copy()

        self.assertTrue(c.storage_id is None)
        self.assertTrue(c._get_name() is None)
        self.assertTrue(c[0,0]==s[0,0])

        c[0,0]=42
        self.assertTrue(c[0,0]!=s[0,0])

        # Clean up
        s.delete_persistent()

    def test_copy_storageNumpyVolatile(self):
        #'''
        #Test that a copy of a StorageNumpy does not share memory (Volatile version)
        #'''
        n=np.arange(12).reshape(3,4)

        s=StorageNumpy(n)
        c=s.copy()

        self.assertTrue(s.storage_id is None)
        self.assertTrue(c.storage_id is None)

        self.assertTrue(c[0,0]==s[0,0])

        c[0,0]=42

        self.assertTrue(c[0,0]!=s[0,0])

    def test_columnar_access(self):
        # Test accessing a column that traverses different blocks in cassandra

        n = np.arange(2*180).reshape(2,180)
        s = StorageNumpy(n, "test_columnar_access")

        s.sync() # Flush values to cassandra
        del s

        s = StorageNumpy(None, "test_columnar_access")

        tmp=s[0,:]

        self.assertTrue(np.array_equal(tmp, n[0,:]))

    def test_row_access(self):
        n = np.arange(64*128).reshape(64,128) # A matrix with "some" columns
        s = StorageNumpy(n, "test_row_access")
        s.sync() # Flush values to cassandra
        del s
        s = StorageNumpy(None, "test_row_access")
        for i in range(0,64):
            tmp = s[i,:]    # Access a whole row
            self.assertTrue(np.array_equal(tmp, n[i,:]))

    def test_column_access(self):
        n = np.arange(2*128).reshape(2,128) # A matrix with "some" columns
        s = StorageNumpy(n, "test_column_access")
        for i in range(0,127):
            tmp = s[:,i]    # Access a whole column
            self.assertTrue(np.array_equal(tmp, n[:,i]))

    def test_slice_after_load(self):
        n = np.arange(2*128).reshape(2,128) # A matrix with "some" columns
        s = StorageNumpy(n, "test_slice_after_load")
        s.sync() # Flush values to cassandra
        del s
        s = StorageNumpy(None, "test_slice_after_load")
        tmp = s[0,110:150]  # Doing an slice on an unloaded numpy
        self.assertTrue(np.array_equal(tmp, n[0,110:150]))

    def test_get_cluster_ids(self):
        n = np.arange(2*128).reshape(2,128) # A matrix with "some" columns
        s = StorageNumpy(n, "test_get_cluster_ids")
        if s._build_args.metas.partition_type != 0: #This test is only valid for ZORDER
            return
        x = s._hcache.get_block_ids(s._build_args.metas)
        # Assuming a BLOCK_SIZE of 4096!! FIXME use an environment variable!
        #print(x)
        self.assertTrue(len(x) == 6)
        #
        #Each element elt of x:
        #    elt[0]==zorderix
        #    elt[1]==cluster_id
        #    elt[2]==block_id
        #    elt[3]==block_coord
        goal=[(0, 0, 0, (0, 0)), (2, 0, 2, (0, 1)), (8, 2, 0, (0, 2)), (10, 2, 2, (0, 3)), (32, 8, 0, (0, 4)), (34, 8, 2, (0, 5))]
        for i,elt in enumerate(x):
            self.assertEqual(elt[0], goal[i][0])
            self.assertEqual(elt[1], goal[i][1])
            self.assertEqual(elt[2], goal[i][2])
            self.assertEqual(elt[3], goal[i][3])

    def test_split(self):
        n = np.arange(2*128).reshape(2,128) # A matrix with "some" columns
        s = StorageNumpy(n, "test_split")
        splits = 0
        for i in s.split():
            # Assuming a BLOCK_SIZE of 4096!! FIXME use an environment variable!
            if splits <= 4:
                self.assertEqual(i.shape, (2,22))
            else:
                self.assertEqual(i.shape, (2,18))
            self.assertTrue(i[0,0] == splits*22)
            splits = splits + 1
        self.assertTrue(splits == 6)

    def test_split_access(self):
        n = np.arange(2*128).reshape(2,128) # A matrix with "some" columns
        s = StorageNumpy(n, "test_split_access")
        if s._build_args.metas.partition_type != 0: #This test is only valid for ZORDER
            return
        splits = 0
        for i in s.split():
            # Assuming a BLOCK_SIZE of 4096!! FIXME use an environment variable!
            if splits <= 4:
                self.assertTrue(np.array_equal(i[:], n[0:22, 22*splits:22*(splits+1)]))
            else:
                self.assertTrue(np.array_equal(i[:], n[0:22, 22*splits:22*(splits)+18]))
            splits = splits + 1

    def test_split_nomem(self):
        n = np.arange(2*128).reshape(2,128) # A matrix with "some" columns
        s = StorageNumpy(n, "test_split_nomem")
        splits = 0
        s.sync() # Flush values to cassandra
        for i in s.split():
            sid = i.storage_id
            i.getID() # Store split in hecuba.istorage
            del i
            i = StorageNumpy(None,None,sid)
            # Assuming a BLOCK_SIZE of 4096!! FIXME use an environment variable!
            if splits <= 4:
                self.assertEqual(i.shape, (2,22))
            else:
                self.assertEqual(i.shape, (2,18))
            self.assertTrue(i[0,0] == splits*22)
            splits = splits + 1
        self.assertTrue(splits == 6)

    def test_split_access_nomem(self):
        n = np.arange(2*128).reshape(2,128) # A matrix with "some" columns
        s = StorageNumpy(n, "test_split_access_nomem")
        if s._build_args.metas.partition_type != 0: #This test is only valid for ZORDER
            return
        s.sync() # Flush values to cassandra
        u = s.storage_id
        splits = 0
        for i in s.split():
            sid = i.storage_id
            i.getID() # Store split in hecuba.istorage
            del i
            i = StorageNumpy(None,None,sid)
            # Assuming a BLOCK_SIZE of 4096!! FIXME use an environment variable!
            if splits <= 4:
                self.assertTrue(np.array_equal(i[:], n[0:22, 22*splits:22*(splits+1)]))
            else:
                self.assertTrue(np.array_equal(i[:], n[0:22, 22*splits:22*(splits)+18]))

            splits = splits + 1

    def test_split_content(self):
        n = np.arange(88*66).reshape(88,66)
        s = StorageNumpy(n,"test_split_content")
        s.sync() # Flush values to cassandra
        del s
        s = StorageNumpy(None,"test_split_content")
        rows = [i for i in s.split(cols=False)]
        self.assertTrue(len(rows)==4)
        columns = [ i for i in s.split(cols=True)]
        self.assertTrue(len(columns)==3)
        blocks = [i for i in s.split()]
        self.assertTrue(len(blocks)==12)
        for i in rows:
            self.assertTrue(i.shape == (22,66))
        for i in columns:
            self.assertTrue(i.shape == (88,22))
        for i in blocks:
            self.assertTrue(i.shape == (22,22))
        self.assertTrue(np.array_equal(rows[0],n[0:22,:]))
        self.assertTrue(np.array_equal(rows[1],n[22:44,:]))
        self.assertTrue(np.array_equal(rows[2],n[44:66,:]))
        self.assertTrue(np.array_equal(rows[3],n[66:,:]))
        self.assertTrue(np.array_equal(columns[0],n[:,0:22]))
        self.assertTrue(np.array_equal(columns[1],n[:,22:44]))
        self.assertTrue(np.array_equal(columns[2],n[:,44:]))

    def test_load_StorageNumpy(self):
        n = np.arange(2*128).reshape(2,128) # A matrix with "some" columns
        s = StorageNumpy(n, "test_load_StorageNumpy")
        s.sync() # Flush values to cassandra
        s2 = StorageNumpy(None, "test_load_StorageNumpy")
        self.assertTrue(s2._is_persistent)
        self.assertEqual(s.storage_id, s2.storage_id)

    def test_np_dot(self):
        n1 = np.arange(8*8).reshape(8,8)
        n2 = np.arange(8*8).reshape(8,8)
        s1 = StorageNumpy(n1, "test_np_dot1")
        s2 = StorageNumpy(n2, "test_np_dot2")
        res = np.dot(s1, s2)
        res.make_persistent("test_np_dots1xs2")
        self.assertTrue(np.array_equal(res, np.dot(n1,n2)))

    @unittest.skip("Only execute for performance reasons")
    def test_performance_storage_numpy_arrow(self):
        # Test the time to retrieve a column from Cassandra

        # Times to repeat the test
        TIMES = 10

        # Matrix sizes to test
        matrix_size = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
        n_cols = 3

        times = {}
        # Test 1 column
        for s in matrix_size:
            times[s] = []  # empty list for size 's'

            # Create a numpy
            n = np.arange(1000*s * n_cols).reshape(1000*s, n_cols)
            matrix_name = "matrix{}x{}".format(1000*s, n_cols)

            # Make it persistent
            o = StorageNumpy(n, matrix_name)

            o.sync() # Flush values to cassandra
            # Clean memory
            del o

            for i in range(TIMES):
                # Retrieve numpy from cassandra (NO data in memory)
                o = StorageNumpy(None, matrix_name)

                # LOAD_ON_DEMAND must be DISABLED!
                self.assertTrue(o.data.hex()[:40], '0' * 40)

                start = timer()

                # Load column
                column = random.randint(0, (n_cols-1))

                o[:, column]

                end = timer()

                # Store time
                times[s].append(end - start)
                del o

        # All tests done, print results
        print("\nRESULTS:")
        for s in matrix_size:
            print("Matrix size{}x{} = ".format(1000*s, n_cols), times[s])
        print("\n")

    def test_setitem_blocks(self):
        # Ensure that sets and gets on different blocks in cassandra are
        # updated and retrieved This test creates a matrix of 3x3 blocks,
        # modifies an element on each of the blocks and retrieves them.
        n = np.arange(66*66).reshape(66,66)
        s = StorageNumpy(n, "test_setitem_blocks")
        magic = [-660 - i for i in range(10)]
        pos = [ (0,0), (0,30), (0,64), (30,0), (30,30), (30,64), (64,0), (64,30), (64,64)]
        # Modify 's' in memory and disk and keep 'n' in the same condition as a baseline
        for i in range(len(pos)):
            s[pos[i]] = magic[i]
            n[pos[i]] = magic[i]
        # Check modified elements in memory
        for i in range(len(pos)):
            self.assertTrue( s[pos[i]] == magic[i] )
        # Check Rest of elements in memory
        self.assertTrue(np.array_equal(n,s))
        s.sync() # Flush values to cassandra
        del s
        s = StorageNumpy(None, "test_setitem_blocks")
        # Check modified elements in Cassandra
        for i in range(len(pos)):
            self.assertTrue( s[pos[i]] == magic[i] )
        # Check Rest of elements in Cassandra
        self.assertTrue(np.array_equal(n,s))

        del s
        s = StorageNumpy(None, "test_setitem_blocks")
        # Modify memory content (not loaded) with different magic values
        for i in range(len(pos)):
            s[pos[i]] = magic[len(pos)-1-i]
            n[pos[i]] = magic[len(pos)-1-i]

        for i in range(len(pos)):
            self.assertTrue( s[pos[i]] == magic[len(pos)-1-i] )
        self.assertTrue(np.array_equal(n,s))

    def test_store_in_view(self):
        n = np.arange(66*66).reshape(66,66)
        s = StorageNumpy(n, "test_store_in_view")

        s1 = s[1:65,1:65]
        s1[0,0] = 666

        self.assertTrue(s[1,1], 666)    # original numpy is modified

        s.sync() # Flush values to cassandra
        del s

        s = StorageNumpy(None, "test_store_in_view")
        self.assertTrue(s[1,1], 666)    # Ensure cassandra has been modified

    # Persistent Views
    def test_pv_slice_slice(self):
        n = np.arange(66*66).reshape(66,66)
        sn = StorageNumpy(n,"test_pv_slice_slice")
        sn.sync() # Flush values to cassandra
        del sn
        sn = StorageNumpy(None,"test_pv_slice_slice")
        # Caso: slice, slice
        s1 = slice(1,65)
        n1 = sn[s1,s1]
        i=1
        j=1
        self.assertTrue(np.array_equal(n1[i,j], n[s1,s1][i,j]))

    def test_loaded(self):
        n = np.arange(88*66).reshape(88,66)
        s = StorageNumpy(n, "test_loaded")
        self.assertTrue(s._numpy_full_loaded is True)
        s.sync() # Flush values to cassandra
        del s
        s = StorageNumpy(None, "test_loaded")
        self.assertTrue(s._numpy_full_loaded is False)

        # The accessed element must be FULL loaded
        row = s[0,:]
        self.assertTrue(s._numpy_full_loaded is False)
        self.assertTrue(row._numpy_full_loaded is True)

        del s
        s = StorageNumpy(None, "test_loaded")
        col = s[:, 0]
        self.assertTrue(s._numpy_full_loaded is False)
        self.assertTrue(col._numpy_full_loaded is True)

        del s
        s = StorageNumpy(None, "test_loaded")
        block = s[22:44, 22:44]
        self.assertTrue(s._numpy_full_loaded is False)
        self.assertTrue(block._numpy_full_loaded is True)

        # Loading ALL elements must make the object full loaded
        del s
        s = StorageNumpy(None, "test_loaded")
        for i in range(s.shape[0]):
            x = s[i,:]
        self.assertTrue(s._numpy_full_loaded is True)

        del s
        s = StorageNumpy(None, "test_loaded")
        for i in range(s.shape[1]):
            x = s[:,i]
        self.assertTrue(s._numpy_full_loaded is True)

        # Split MUST NOT load the object
        del s
        s = StorageNumpy(None, "test_loaded")
        rows = [ i for i in s.split(cols=False) ]
        for i in rows:
            self.assertTrue(i._numpy_full_loaded is False)

        del s
        s = StorageNumpy(None, "test_loaded")
        columns = [ i for i in s.split(cols=True) ]
        for i in columns:
            self.assertTrue(i._numpy_full_loaded is False)

        del s
        s = StorageNumpy(None, "test_loaded")
        blocks = [ i for i in s.split() ]
        for i in blocks:
            self.assertTrue(i._numpy_full_loaded is False)


    # TODO: Tranform SNadaptcoords.py
    def test_out_of_bounds_in_numpy(self):
        n = np.arange(88*66).reshape(88,66)
        s = StorageNumpy(n, "test_bounds_in_numpy")
        del s
        s = StorageNumpy(None, "test_bounds_in_numpy")

        with self.assertRaises(IndexError):
            s[:, 100]

        with self.assertRaises(IndexError):
            s[100, :]

        v = s[1:10,22:50]
        with self.assertRaises(IndexError):
            v[11, :]
        with self.assertRaises(IndexError):
            v[:, 55]

    def views_with_steps(self):
        n = np.arange(88*66).reshape(88,66)
        s = StorageNumpy(n, "views_with_steps")

        self.assertEqual(self._row_elem, 22) # HARDCODED VALUE!

        self.assertEqual(s._n_blocks, 12)

        v1 = s[:,23:40]
        self.assertEqual(v1._n_blocks, 4)

        v = s[:,2:50:2]
        self.assertEqual(v._n_blocks, 12)

        v2 = s[:, 23:50:2]  # 23/2 == 11 columns
        self.assertEqual(v2._n_blocks, 8)

    def test_sync(self):
        n = np.arange(22*22).reshape(22,22)
        s = StorageNumpy(n, "test_sync")

        del s
        s = StorageNumpy(None, "test_sync")
        s[0,0] = 666    # Asynchronous write
        x = StorageNumpy(None, None, s.storage_id)
        self.assertTrue(s[0,0] != x[0,0]) # Data is still in dirty
        self.assertTrue(x[0,0] == 0)
        s.sync()
        x = StorageNumpy(None, None, s.storage_id)
        self.assertTrue(s[0,0] == x[0,0])
        self.assertTrue(x[0,0] == 666)

    # Persistent Views: slice, int
    def test_pv_slice_int(self):
        n = np.arange(66*66).reshape(66,66)
        sn = StorageNumpy(n,"test_pv_slice_int")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_slice_int")
        # Caso: slice, int
        s1 = slice(1,65)
        s2 = 30
        i=1
        n2 = sn[s1,s2]
        self.assertTrue(np.array_equal(n2[i], n[s1,s2][i]))

    # Persistent Views: int, slice
    def test_pv_int_slice(self):
        n = np.arange(66*66).reshape(66,66)
        sn = StorageNumpy(n,"test_pv_int_slice")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_int_slice")
        # Caso: int, slice
        s1 = slice(1,65)
        s2 = 30
        i=1
        n2 = sn[s2,s1]
        self.assertTrue(np.array_equal(n2[i], n[s2,s1][i]))

    # Persistent Views: slice_step
    def test_pv_slice_step(self):
        n = np.arange(66*66).reshape(66,66)
        sn = StorageNumpy(n,"test_pv_slice_step")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_slice_step")
        s1 = slice(1,65,2)
        n2 = sn[s1,s1]
        i=2
        j=30
        self.assertTrue(np.array_equal(n[s1,s1][i,j], n2[i,j]))

    # Persistent Views: slice_from_slice
    def test_pv_slice_from_slice(self):
        n = np.arange(66*66).reshape(66,66)
        sn = StorageNumpy(n,"test_pv_slice_from_slice")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_slice_from_slice")
        s1 = slice(1,65)
        s2 = slice(1,20)
        n1 = sn[s1]
        n2 = n1[s2]
        self.assertTrue(np.array_equal(n2, n[s1][s2]))

    # Persistent Views: slice_from_slice_step
    def test_pv_slice_from_from_slice_step(self):
        n = np.arange(66*66).reshape(66,66)
        sn = StorageNumpy(n,"test_pv_slice_from_slice_step")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_slice_from_slice_step")
        s1 = slice(1,65,2)
        s2 = slice(1,20,2)
        n1 = sn[s1][s2]
        self.assertTrue(np.array_equal(n1, n[s1][s2]))

    # Persistent Views: multiple slices
    def test_pv_slice_from_N_slice(self):
        n = np.arange(66*66).reshape(66,66)
        sn = StorageNumpy(n,"test_pv_slice_from_N_slice")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_slice_from_N_slice")
        s1 = slice(1,65,2)
        s2 = slice(1,20,2)
        n1 = sn[s1,s1][s2,s2]
        self.assertTrue(np.array_equal(n1, n[s1,s1][s2,s2]))

    # Persistent Views: only_int
    def test_pv_only_int(self):
        n = np.arange(66*66).reshape(66,66)
        sn = StorageNumpy(n,"test_pv_only_int")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_only_int")
        s1 = 1
        n1 = sn[s1]
        self.assertTrue(np.array_equal(n[1], n1))

    # Persistent Views: big_np
    def test_pv_big_np(self):
        n = np.arange(1000*1000).reshape(1000,1000)
        sn = StorageNumpy(n,"test_pv_big_np")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_big_np")
        s1 = (22,22)
        self.assertTrue(np.array_equal(sn[s1], n[s1]))

    # Persistent Views: one_dim
    def test_pv_one_dim(self):
        n = np.arange(66*66)
        sn = StorageNumpy(n,"test_pv_one_dim")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_one_dim")
        s1 = 30
        self.assertTrue(np.array_equal(sn[s1], n[s1]))

    # Persistent Views: negative_indexes
    def test_pv_negative_indexes(self):
        n = np.arange(66*66).reshape(66,66)
        sn = StorageNumpy(n,"test_pv_negative_indexes")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_negative_indexes")
        s1 = -1
        self.assertTrue(np.array_equal(sn[s1], n[s1]))
        nn = np.arange(66*66)
        snn = StorageNumpy(nn,"test_pv_negative_indexes_small")
        del snn
        snn = StorageNumpy(None,"test_pv_negative_indexes_small")
        self.assertTrue(np.array_equal(snn[s1], nn[s1]))

    # Persistent Views: special_case 
    def test_pv_special_case(self):
        n = np.arange(66*66).reshape(66,66)
        sn = StorageNumpy(n,"test_pv_special_case")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_special_case")
        s1 = slice(1,65)
        s2 = sn[s1,s1]
        ssf=1
        self.assertTrue(np.array_equal(sn[s1,s1][1], n[s1,s1][1]))

    # Persistent Views: slice_single_row 
    def test_pv_slice_single_row(self):
        n = np.arange(66*66).reshape(66,66)
        sn = StorageNumpy(n,"test_pv_slice_single_row")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_slice_single_row")
        s1 = slice(1,65)
        s2 = slice(1, None, None)
        self.assertTrue(np.array_equal(sn[s1], n[s1]))
        self.assertTrue(np.array_equal(sn[s1][s2], n[s1][s2]))
        self.assertTrue(np.array_equal(sn[s1][s2][s2], n[s1][s2][s2]))

    # Persistent Views: load_correct_blocks 
    def test_pv_load_correct_blocks(self):
        n = np.arange(66*66).reshape(66,66)
        sn = StorageNumpy(n,"test_pv_load_correct_blocks")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_load_correct_blocks")
        s1 = (0, slice(None, None, None))
        x = sn[s1]
        self.assertTrue(len(sn._loaded_coordinates) == 3)

    # Persistent Views: slice_single_column 
    def test_pv_slice_single_column(self):
        n = np.arange(66*66).reshape(66,66)
        sn = StorageNumpy(n,"test_pv_slice_single_column")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_slice_single_column")
        s1 = (slice(None, None, None), 30)
        s2 = slice(1, None, None)
        self.assertTrue(sn[s1].shape == n[s1].shape)
        self.assertTrue(np.array_equal(sn[s1], n[s1]))
        self.assertTrue(np.array_equal(sn[s1][s2], n[s1][s2]))
        self.assertTrue(np.array_equal(sn[s1][s2][s2], n[s1][s2][s2]))

    # Persistent Views: three_dimensions 
    def test_pv_three_dimensions(self):
        n = np.arange(3*66*66).reshape(3,66,66)
        sn = StorageNumpy(n,"test_pv_three_dimensions")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_three_dimensions")
        s1 = (0, 1, slice(None, None, None))
        self.assertTrue(np.array_equal(sn[s1], n[s1]))
        s2 = slice(1,10,1)
        self.assertTrue(np.array_equal(sn[s1][s2], n[s1][s2]))

    # Persistent Views: three_dimensions_easy 
    def test_pv_three_dimensions_easy(self):
        n = np.arange(4*4*4).reshape(4,4,4)
        sn = StorageNumpy(n,"test_pv_three_dimensions_easy")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_three_dimensions_easy")
        orig3 = (slice(None, None, None), slice(None, None, None), slice(None, None, None))
        s1 = (0, 1, slice(None, None, None))
        self.assertTrue(np.array_equal(sn[s1], n[s1]))
        s2 = slice(1, None, None)
        self.assertTrue(np.array_equal(sn[s1][s2], n[s1][s2]))
        self.assertTrue(np.array_equal(sn[s1][s2][s2], n[s1][s2][s2]))

    # Persistent Views: three_dimensions_all_coords 
    def test_pv_three_dimensions_all_coords(self):
        n = np.arange(8*8*8).reshape(8,8,8)
        sn = StorageNumpy(n,"test_pv_three_dimensions_all_coords")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_three_dimensions_all_coords")
        orig3 = (slice(None, None, None), slice(None, None, None), slice(None, None, None))
        coords = []
        for i in sn.calculate_block_coords(orig3):
            coords.append(i)
        expected = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
        result = all(map(lambda x, y: x == y, expected, coords))
        self.assertTrue(result, True)

    # Persistent Views: three_dimensions_subslice_onedim_coords 
    def test_pv_three_dimensions_slice_onedim(self):
        n = np.arange(8*8*8).reshape(8,8,8)
        sn = StorageNumpy(n,"test_pv_three_dimensions_slice_onedim")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_three_dimensions_slice_onedim")
        ss = sn[0,0,slice(None,None,None)]
        coords = []
        for i in ss.calculate_block_coords(ss._build_args.view_serialization):
            coords.append(i)
        expected = [(0, 0, 0), (0, 0, 1)]
        result = all(map(lambda x, y: x == y, expected, coords))
        self.assertTrue(result, True)

    # Persistent Views: three_dimensions_subslice_twodim_coords 
    def test_pv_three_dimensions_slice_twodim(self):
        n = np.arange(8*8*8).reshape(8,8,8)
        sn = StorageNumpy(n,"test_pv_three_dimensions_slice_twodim")
        sn.sync()
        del sn
        sn = StorageNumpy(None,"test_pv_three_dimensions_slice_twodim")
        ss = sn[(slice(None,None,None), slice(None,None,None),0)]
        coords = []
        for i in ss.calculate_block_coords(ss._build_args.view_serialization):
            coords.append(i)
        expected = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
        result = all(map(lambda x, y: x == y, expected, coords))
        self.assertTrue(result, True)

    # Persistent Views: two_dimensions_multiple_accesses 
    def test_pv_two_dimensions_multiple_accesses(self):
        n = np.arange(66*66).reshape(66,66)
        s = StorageNumpy(n,"test_pv_two_dimensions_multiple_accesses")
        s.sync()
        del s
        s = StorageNumpy(None,"test_pv_two_dimensions_multiple_accesses")
        s1 = (slice(None, None, None), slice(2,50,2)) # slice with step != 1
        x = s[s1]
        res = x.shape == n[s1].shape
        self.assertTrue(np.array_equal( x, n[x._build_args.view_serialization] ))
        self.assertTrue(x[0,0] == n[x._build_args.view_serialization][0,0])
        self.assertTrue(x[0,1] == n[x._build_args.view_serialization][0,1])
        self.assertTrue(x[0,-2] == n[x._build_args.view_serialization][0,-2])
        self.assertTrue(x[0,-1] == n[x._build_args.view_serialization][0,-1])
        self.assertTrue(x[-2,0] == n[x._build_args.view_serialization][-2,0])
        self.assertTrue(x[-1,0] == n[x._build_args.view_serialization][-1,0])
        self.assertTrue(x[-2,-2] == n[x._build_args.view_serialization][-2,-2])
        self.assertTrue(x[-1,-1] == n[x._build_args.view_serialization][-1,-1])

        del s
        s = StorageNumpy(None,"test_pv_two_dimensions_multiple_accesses")
        s1 = (slice(None, None, None), slice(2,50,2))
        x = s[s1]
        s2 = (slice(None, None,None), slice(-10, -3, 1))
        self.assertTrue(np.array_equal(x[s2], n[x._build_args.view_serialization][s2]))

        del s
        s = StorageNumpy(None,"test_pv_two_dimensions_multiple_accesses")
        s1 = (slice(None, None, None), slice(2,50,2))
        x = s[s1]
        s2 = (slice(None,None,None),slice(-3, -10, 1))
        self.assertTrue(np.array_equal(x[s2], n[x._build_args.view_serialization][s2]))

        del s
        s = StorageNumpy(None,"test_pv_two_dimensions_multiple_accesses")
        s1 = (slice(None, None, None), slice(2,50,2))
        x = s[s1]
        s2 = (slice(None,None,None),slice(-10, 200, 1))
        self.assertTrue(np.array_equal(x[s2], n[x._build_args.view_serialization][s2]))

        del s
        s = StorageNumpy(None,"test_pv_two_dimensions_multiple_accesses")
        s1 = (slice(None, None, None), slice(2,50,2))
        x = s[s1]
        s2 = (slice(None,None,None),slice(2, -2, 1))
        self.assertTrue(np.array_equal(x[s2], n[x._build_args.view_serialization][s2]))

    # Simple_negative
    def test_simple_negative(self):
        n = np.arange(6)
        s = StorageNumpy(n,"test_simple_negative")
        ss = s[2::2]
        self.assertTrue(ss[-1] == n[2::2][-1])

    # This code tries to access an out of bounds array
    def test_out_of_bounds(self):
        n = np.arange(1000).reshape(10,10,10)
        coordinates = (slice(50, 150, None), slice(50, 150, None), slice(5, 150, None))
        s = StorageNumpy(n, "KK")
        t  = s[coordinates]
        t - 1   # Should not fail
        t-= 1   # Should not fail

    def test_arrow_access(self):
        n = np.arange(50*50).reshape(50,50)
        s = StorageNumpy(n, "test_arrow_access")
        s.sync()
        del s
        s = StorageNumpy(None, "test_arrow_access")
        x = s[:, 20]
        self.assertTrue(np.array_equal(x, n[:,20]))
        y = s[:, 30]
        self.assertTrue(np.array_equal(y, n[:,30]))
        z = s[:, 49]
        self.assertTrue(np.array_equal(z, n[:,49]))


if __name__ == '__main__':
    unittest.main()
