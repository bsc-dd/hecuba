import unittest
import gc

from hecuba import config, StorageNumpy
import uuid
import numpy as np


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
        storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, tablename + typecode)
        niter = 2

        for _ in range(niter):
            # Build array and store
            typed_array = StorageNumpy(base_array, storage_id, tablename)
            self.assertTrue(np.array_equal(typed_array, base_array))

            del typed_array
            gc.collect()

            # Load array
            typed_array = StorageNumpy(None, storage_id, tablename)
            self.assertTrue(np.allclose(typed_array, base_array))
            typed_array.delete_persistent()

    def test_types_persistence(self):
        base_array = np.arange(256)
        tablename = self.ksp + '.' + self.table

        for typecode in np.typecodes['Integer']:
            if typecode == 'p':
                # TODO For now skip arrays made of pointers
                pass
            storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, tablename + typecode)
            typed_array = StorageNumpy(base_array.astype(typecode), storage_id, tablename)
            self.assertTrue(np.array_equal(typed_array, base_array.astype(typecode)))

            del typed_array
            gc.collect()

            typed_array = StorageNumpy(None, storage_id, tablename)
            self.assertTrue(np.allclose(typed_array, base_array.astype(typecode)))
            typed_array.delete_persistent()

        for typecode in np.typecodes['UnsignedInteger']:
            if typecode == 'P':
                # TODO For now skip arrays made of pointers
                pass
            storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, tablename + typecode)
            typed_array = StorageNumpy(base_array.astype(typecode), storage_id, tablename)
            self.assertTrue(np.allclose(typed_array, base_array.astype(typecode)))

            del typed_array
            gc.collect()

            typed_array = StorageNumpy(None, storage_id, tablename)
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

        storage_id = uuid.uuid4()
        base_array = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(input_array=base_array, name="testing_arrays.first_test", storage_id=storage_id)

        del casted
        gc.collect()

        test_numpy = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(name="testing_arrays.first_test", storage_id=storage_id)
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

        storage_id = uuid.uuid4()
        base_array = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(input_array=base_array, name="testing_arrays.first_test", storage_id=storage_id)

        del casted
        gc.collect()
        test_numpy = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(name="testing_arrays.first_test", storage_id=storage_id)
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

    def test_slicing_3d(self):
        base = np.arange(8 * 8 * 4).reshape((8, 8, 4))
        hecu = StorageNumpy(input_array=base, name='my_array')
        storage_id = hecu.storage_id
        res_hecu = hecu[6:7, 4:]
        res = base[6:7, 4:]
        self.assertTrue(np.array_equal(res, res_hecu))


        del hecu
        del res_hecu
        gc.collect()

        hecu = StorageNumpy(storage_id=storage_id)
        res_hecu = hecu[6:7, 4:]
        self.assertTrue(np.array_equal(res, res_hecu))

        hecu.delete_persistent()

    def test_slicing_ndims(self):
        import random
        ndims = 10
        max_elements = 2048
        for dims in range(1, ndims):
            storage_id = uuid.uuid4()
            elem_per_dim = int(max_elements **(1/dims))
            select = (slice(random.randint(0, elem_per_dim)),)*dims
            base = np.arange(elem_per_dim**dims).reshape((elem_per_dim, )*dims)

            hecu = StorageNumpy(input_array=base, name='my_array', storage_id=storage_id)
            res_hecu = hecu[select]
            res = base[select]
            self.assertTrue(np.array_equal(res, res_hecu))

            del hecu
            gc.collect()

            hecu = StorageNumpy(storage_id=storage_id)
            res_hecu = hecu[select]
            res = base[select]
            self.assertTrue(np.array_equal(res, res_hecu))
            hecu.delete_persistent()
            del res_hecu
            del hecu



if __name__ == '__main__':
    unittest.main()
