import unittest

from hecuba import config, StorageNumpy, StorageObj
import uuid
import numpy as np

class TestStorageObjNumpy(StorageObj):
    '''
       @ClassField mynumpy numpy.ndarray
    '''
    pass

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

    def test_numpy_reserved(self):
        size = 40000
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        base_array = np.arange(40000)
        no.mynumpy = base_array
        tablename = "my_app.numpy_test_%d" % size
        typed_array = None
        for typecode in np.typecodes['Integer']:
            if typecode == 'p':
                # TODO For now skip arrays made of pointers
                pass
            storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, tablename)
            coord = [slice(20000, 21000, None), slice(22000, 35001, None)]
            typed_array = StorageNumpy(base_array.astype(typecode), storage_id, tablename, coord, False)
        a = None

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
            typed_array = None
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


if __name__ == '__main__':
    unittest.main()
