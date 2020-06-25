import gc
import unittest

from hecuba import config, StorageNumpy
import uuid
import numpy as np

from storageAPI.storage.api import getByID


class StorageNumpyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = config.execution_name
        config.NUM_TEST = 0 # HACK a new attribute to have a global counter
    @classmethod
    def tearDownClass(cls):
        config.execution_name = cls.old
        del config.NUM_TEST

    # Create a new keyspace per test
    def setUp(self):
        config.NUM_TEST = config.NUM_TEST + 1
        self.ksp = "StorageNumpyTest{}".format(config.NUM_TEST).lower()
        config.execution_name = self.ksp

    def tearDown(self):
        config.session.execute("DROP KEYSPACE IF EXISTS {}".format(self.ksp))

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
        tablename = self.ksp + '.' + self.table

        typecode = 'mytype'
        niter = 2

        for _ in range(niter):
            # Build array and store
            typed_array = StorageNumpy(base_array, tablename)
            self.assertTrue(np.array_equal(typed_array, base_array))

            # Load array
            typed_array = StorageNumpy(None, tablename)
            self.assertTrue(np.allclose(typed_array, base_array))
            typed_array.delete_persistent()

    def test_types_persistence(self):
        base_array = np.arange(256)
        tablename = self.ksp + '.' + self.table

        for typecode in np.typecodes['Integer']:
            if typecode == 'p':
                # TODO For now skip arrays made of pointers
                pass
            typed_array = StorageNumpy(base_array.astype(typecode), tablename)
            self.assertTrue(np.array_equal(typed_array, base_array.astype(typecode)))

            typed_array = StorageNumpy(None, tablename)
            self.assertTrue(np.allclose(typed_array, base_array.astype(typecode)))
            typed_array.delete_persistent()

        for typecode in np.typecodes['UnsignedInteger']:
            if typecode == 'P':
                # TODO For now skip arrays made of pointers
                pass
            typed_array = StorageNumpy(base_array.astype(typecode), tablename)
            self.assertTrue(np.allclose(typed_array, base_array.astype(typecode)))

            typed_array = StorageNumpy(None, tablename)
            self.assertTrue(np.allclose(typed_array, base_array.astype(typecode)))
            typed_array.delete_persistent()

    def test_read_all(self):
        nelem = 2 ** 21
        elem_dim = 2 ** 7

        base_array = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(input_array=base_array, name="first_test")

        test_numpy = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(name="first_test")
        chunk = casted[slice(None, None, None)]
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))
        casted.delete_persistent()

    def test_numpy_reserved_5d_read_all(self):

        nelem = 100000
        elem_dim = 10

        base_array = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(input_array=base_array, name="first_test")

        test_numpy = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(name="first_test")
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
        hecu_p = StorageNumpy(input_array=base, name='my_array')
        hecu_r2 = StorageNumpy(name="my_array")
        res = hecu_r2[:3, :2]
        sum = res.sum()
        res = hecu_r2[:3, :2]
        avg = res.mean()
        self.assertGreater(sum, 0)
        self.assertGreater(avg, 0)

    def test_slicing_3d(self):
        base = np.arange(8 * 8 * 4).reshape((8, 8, 4))
        hecu = StorageNumpy(input_array=base, name='my_array')
        res_hecu = hecu[6:7, 4:]
        res = base[6:7, 4:]
        self.assertTrue(np.array_equal(res, res_hecu))

        hecu = StorageNumpy(name="my_array")
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

            hecu = StorageNumpy(input_array=base, name='my_array')
            res_hecu = hecu[select]
            res = base[select]
            self.assertTrue(np.array_equal(res, res_hecu))

            hecu = StorageNumpy(name="my_array")
            res_hecu = hecu[select]
            res = base[select]
            self.assertTrue(np.array_equal(res, res_hecu))
            hecu.delete_persistent()
            del res_hecu
            del hecu

    def test_slice_ops(self):
        obj = np.arange(8 * 8 * 8).reshape((8, 8, 8))
        hecu = StorageNumpy(input_array=obj, name='some_name')
        hecu_sub = hecu[:2, 3:, 4:]
        sum = hecu_sub.sum()
        self.assertGreater(sum, 0)
        description = repr(hecu_sub)
        self.assertIsInstance(description, str)
        hecu.delete_persistent()

    def test_slice_ops2(self):
        obj = np.arange(8 * 8 * 8).reshape((8, 8, 8))
        hecu = StorageNumpy(input_array=obj, name='some_name')
        hecu_sub = hecu[:2, 3:, 4:]
        hecu_sub2 = hecu_sub[:1, 2:, 3:]
        sum = hecu_sub2.sum()
        self.assertGreater(sum, 0)
        description = repr(hecu_sub2)
        self.assertIsInstance(description, str)
        hecu.delete_persistent()

    def test_slice_from_numpy_array(self):
        obj = np.arange(8 * 8 * 8).reshape((8, 8, 8))
        hecu = StorageNumpy(input_array=obj, name='some_name')
        l = np.array((0,1))
        hecu_sub = hecu[l]  #Access using an array of indexes
# FIXME add more testing, currently if it does not segfault, then it works
#        sum = hecu_sub.sum()
#        self.assertEqual(sum, obj[l].sum())
        hecu.delete_persistent()

    def test_iter_numpy(self):
        obj = np.arange(8 * 8 * 8).reshape((8, 8, 8))
        hecu = StorageNumpy(input_array=obj, name='some_name')
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
        hecu_p = StorageNumpy(input_array=base, name='my_array')
        sub_hecu = hecu_p[:2, 3:]
        sub_hecu[0][2:] = 0
        hecu_p_load = StorageNumpy(name="my_array")
        rep = repr(hecu_p_load)
        self.assertIsInstance(rep, str)
        load_sub_arr = hecu_p_load[:]
        self.assertTrue(np.array_equal(load_sub_arr, np.arange(8 * 8 * 4).reshape((8, 8, 4))))
        hecu_p_load.delete_persistent()

    def test_assign_element(self):
        base = np.arange(8 * 8 * 4).reshape((8, 8, 4))
        hecu_p = StorageNumpy(input_array=base, name='my_array2')
        sub_hecu = hecu_p[:2, 3:]
        sub_hecu[0][1][0] = 0
        hecu_p_load = StorageNumpy(name="my_array2")
        rep = repr(hecu_p_load)
        self.assertIsInstance(rep, str)
        load_sub_arr = hecu_p_load[:]
        self.assertTrue(np.array_equal(load_sub_arr, np.arange(8 * 8 * 4).reshape((8, 8, 4))))
        hecu_p_load.delete_persistent()

    def test_load_2_dif_clusters_same_instance(self):
        base = np.arange(50 * 50).reshape((50, 50))
        hecu_p = StorageNumpy(input_array=base, name='my_array3')
        hecu_p_load = StorageNumpy(name="my_array3")
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

        data = StorageNumpy(input_array=x, name="test_array")

        for i, chunk in enumerate(data.np_split(block_size=(bn, bm))):
            storage_id = chunk.storage_id
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

        data = StorageNumpy(input_array=x, name="test_array")

        for i, chunk in enumerate(data.np_split(block_size=(bn, bm))):
            storage_id = chunk.storage_id
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

        data = StorageNumpy(input_array=x, name="test_array")

        for i, chunk in enumerate(data.np_split(block_size=(bn, bm))):
            storage_id = chunk.storage_id
            del chunk
            chunk = getByID(storage_id)
            self.assertTrue(np.array_equal(list(chunk), blocks[i]))

        self.assertEqual(i + 1, len(blocks))

    def test_split_already_persistent(self):

        bn, bm = (2, 1)
        x = np.arange(100).reshape(10, -1)
        blocks = []
        for i in range(0, x.shape[0], bn):
            row = [x[i: i + bn, j: j + bm] for j in range(0, x.shape[1], bm)]
            blocks.append(row)

        data = StorageNumpy(input_array=x, name="test_array")

        for i, chunk in enumerate(data.np_split(block_size=(bn, bm))):
            storage_id = chunk.storage_id
            del chunk
            chunk = getByID(storage_id)
            self.assertTrue(np.array_equal(list(chunk), blocks[i]))

        del data
        gc.collect()

        data = StorageNumpy(name="test_array")
        self.assertTrue(np.array_equal(list(data), x))

        for i, chunk in enumerate(data.np_split(block_size=(bn, bm))):
            storage_id = chunk.storage_id
            del chunk
            chunk = getByID(storage_id)
            self.assertTrue(np.array_equal(list(chunk), blocks[i]))

        self.assertEqual(i + 1, len(blocks))

    def test_storagenumpy_copy_memory(self):
        '''
        Check that the memory from a StorageNumpy does not share original array
        '''
        n = np.arange(12).reshape(3,4)

        s1 = StorageNumpy(n, "test_storage_copy_memory")

        # StorageNumpy s1 and n should NOT share memory
        s1[0][0] = 42
        self.assertTrue(not np.array_equal(s1, n))
        s1[0][0] = n[0][0] # Undo

        n[2][2] = 666
        self.assertTrue(not np.array_equal(s1, n))
        # Clean up
        s1.delete_persistent()
        config.session.execute("DROP TABLE IF EXISTS my_app.test_storage_copy_memory")


    def test_storagenumpy_from_storagenumpy(self):
        '''
        Create a StorageNumpy from another StorageNumpy
        '''

        n = np.arange(12).reshape(3,4)

        s1 = StorageNumpy(n, "test_storage_from_storage")

        s2 = StorageNumpy(s1) # Create a StorageNumpy from another StorageNumpy

        self.assertTrue(s2.storage_id != s1.storage_id)
        self.assertTrue(s2._get_name() == s1._get_name())
        self.assertTrue(np.array_equal(s2, n))

        # StorageNumpy s1 and s2 should share memory
        s1[0][0] = 42
        self.assertTrue(np.array_equal(s2, s1))

        s2[2][2] = 666
        self.assertTrue(np.array_equal(s2, s1))

        # Create a third StorageNumpy
        s3 = StorageNumpy(s2)

        self.assertTrue(s3.storage_id != s2.storage_id)
        self.assertTrue(s3._get_name() == s3._get_name())
        self.assertTrue(np.array_equal(s3, s2))

        # Clean up
        s1.delete_persistent()
        s2.delete_persistent()
        s3.delete_persistent()
        config.session.execute("DROP TABLE IF EXISTS my_app.test_storage_from_storage")

    def test_storagenumpy_reshape(self):
        '''
        Reshape a StorageNumpy
        '''

        n = np.arange(12).reshape(3,4)

        s1 = StorageNumpy(n, "test_storagenumpy_reshape")

        r = s1.reshape(4,3)
        self.assertTrue(r.storage_id != s1.storage_id)
        self.assertTrue(r.shape != s1.shape)
        self.assertTrue(r.strides != s1.strides)


        # Clean up
        s1.delete_persistent()
        config.session.execute("DROP TABLE IF EXISTS my_app.test_storage_from_storage")



if __name__ == '__main__':
    unittest.main()
