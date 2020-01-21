import unittest

from hecuba import config, StorageNumpy
import uuid
import numpy as np

from storageAPI.storage.api import getByID


class StorageNumpyTest(unittest.TestCase):
    table = 'numpy_test'
    ksp = 'my_app'

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
        from cassandra import InvalidRequest
        try:
            config.session.execute("TRUNCATE TABLE testing_arrays.first_test;")
        except InvalidRequest:
            pass
        try:
            config.session.execute("TRUNCATE TABLE testing_arrays.first_test_numpies;")
        except InvalidRequest:
            pass

        nelem = 2 ** 21
        elem_dim = 2 ** 7

        base_array = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(input_array=base_array, name="testing_arrays.first_test")

        test_numpy = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(name="testing_arrays.first_test")
        chunk = casted[slice(None, None, None)]
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))
        casted.delete_persistent()

    def test_numpy_reserved_5d_read_all(self):
        from cassandra import InvalidRequest
        try:
            config.session.execute("TRUNCATE TABLE testing_arrays.first_test;")
        except InvalidRequest:
            pass
        try:
            config.session.execute("TRUNCATE TABLE testing_arrays.first_test_numpies;")
        except InvalidRequest:
            pass

        nelem = 100000
        elem_dim = 10

        base_array = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(input_array=base_array, name="testing_arrays.first_test")

        test_numpy = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(name="testing_arrays.first_test")
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
        config.session.execute("DROP TABLE IF EXISTS hecuba_dislib.test_array")
        config.session.execute("DROP TABLE IF EXISTS hecuba_dislib.test_array_numpies")
        config.session.execute("TRUNCATE TABLE hecuba.istorage")

        block_size = (20, 10)
        x = np.array([[i] * 10 for i in range(100)])
        storage_id = uuid.uuid4()

        data = StorageNumpy(input_array=x, name="hecuba_dislib.test_array", storage_id=storage_id)

        for i, chunk in enumerate(data.np_split(block_size=block_size[0], axis="rows")):
            r_x = np.array([[j] * 10 for j in range(i * block_size[0], i * block_size[0] + block_size[0])])

            storage_id = chunk.storage_id
            del chunk
            chunk = getByID(storage_id)
            self.assertTrue(np.array_equal(list(chunk), r_x))

        self.assertEqual(i + 1, len(data) // block_size[0])


if __name__ == '__main__':
    unittest.main()
