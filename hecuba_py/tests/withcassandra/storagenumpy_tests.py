import unittest

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

    def test_types(self):
        base_array = np.arange(256)
        storage_id = None
        tablename = None

        for typecode in np.typecodes['Integer']:
            typed_array = StorageNumpy(base_array.astype(typecode), storage_id, tablename)
            self.assertTrue(np.array_equal(typed_array, base_array.astype(typecode)))

        for typecode in np.typecodes['UnsignedInteger']:
            typed_array = StorageNumpy(base_array.astype(typecode), storage_id, tablename)
            self.assertTrue(np.array_equal(typed_array, base_array.astype(typecode)))

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
            import gc
            del typed_array
            gc.collect()
            typed_array = StorageNumpy(None, storage_id, tablename)
            self.assertTrue(np.array_equal(typed_array, base_array.astype(typecode)))

        for typecode in np.typecodes['UnsignedInteger']:
            if typecode == 'P':
                # TODO For now skip arrays made of pointers
                pass
            storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, tablename + typecode)
            typed_array = StorageNumpy(base_array.astype(typecode), storage_id, tablename)
            self.assertTrue(np.array_equal(typed_array, base_array.astype(typecode)))
            typed_array = None
            typed_array = StorageNumpy(None, storage_id, tablename)
            self.assertTrue(np.array_equal(typed_array, base_array.astype(typecode)))

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

    def test_read_all(self):
        nelem = 2 ** 21
        elem_dim = 2 ** 7

        storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, "first_test")
        base_array = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(input_array=base_array, name="testing_arrays.first_test", storage_id=storage_id)
        import gc
        del casted
        gc.collect()
        test_numpy = np.arange(nelem).reshape((elem_dim, elem_dim, elem_dim))
        casted = StorageNumpy(name="testing_arrays.first_test", storage_id=storage_id)
        chunk = casted[slice(None, None, None)]
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))


if __name__ == '__main__':
    unittest.main()
