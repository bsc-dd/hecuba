import unittest

import numpy as np
import uuid
from hecuba import config, StorageNumpy, StorageObj


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

    def test_numpy_reserved_all_cluster_2D(self):
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_0;")
        size = 0
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        no.mynumpy = np.arange(40000).reshape((200, 200))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        chunk = myobj2.mynumpy  # we are getting the first cluster

    def test_numpy_reserved_2d_1cluster(self):
        coordinates = [slice(0, 50, None), slice(0, 50, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_1;")
        size = 1
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        self.assertEqual(result[0, 43], 43)
        self.assertEqual(result[43, 43], 4343)
        self.assertNotEqual(result[0, 44], 4400)
        self.assertNotEqual(result[45, 45], 4544)

    def test_numpy_reserved_2d_2cluster(self):
        coordinates = [slice(0, 22, None), slice(0, 44, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_10;")
        size = 10
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        chunk = myobj2.mynumpy[coordinates]  # we are getting the first cluster
        result = chunk.view(np.ndarray)
        self.assertEqual(result[21, 0], 2100)

    def test_numpy_reserved_2d_3cluster(self):
        coordinates = [slice(0, 45, None), slice(0, 22, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_100;")
        size = 100
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        chunk = myobj2.mynumpy[coordinates]  # we are getting the first cluster
        result = chunk.view(np.ndarray)
        self.assertEqual(result[44, 0], 4400)

    def test_numpy_reserved_2d_4cluster(self):
        coordinates = [slice(0, 45, None), slice(0, 45, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_1000;")
        size = 1000
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        chunk = myobj2.mynumpy[coordinates]# we are getting the first cluster
        result = chunk.view(np.ndarray)
        self.assertNotEqual(result[43, 43], 4443)
        self.assertEqual(result[44, 44], 4444)

    def test_numpy_reserved_3d_1cluster(self):
        coordinates = [slice(0, 45, None), slice(0, 45, None), slice(0, 45, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_10000;")
        size = 10000
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        no.mynumpy = np.arange(1000).reshape((10, 10, 10))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        chunk = myobj2.mynumpy[coordinates]  # we are getting the first cluster
        result = chunk.view(np.ndarray)
        self.assertEqual(result[5, 5, 5], 555)

    def test_numpy_reserved_4d_1cluster(self):
        coordinates = [slice(0, 45, None), slice(0, 45, None), slice(0, 45, None), slice(0, 45, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_100000;")
        size = 100000
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        no.mynumpy = np.arange(10000).reshape((10, 10, 10, 10))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        chunk = myobj2.mynumpy[coordinates]  # we are getting the first cluster
        result = chunk.view(np.ndarray)
        self.assertEqual(result[5, 5, 5, 5], 5555)

    def test_types_persistence(self):
        base_array = np.arange(256)
        tablename = self.ksp + '.' + self.table

        for typecode in np.typecodes['Integer']:
            if typecode == 'p':
                # TODO For now skip arrays made of pointers
                pass
            storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, tablename + typecode)
            typed_array = StorageNumpy(base_array.astype(typecode), storage_id, tablename)
            self.assertTrue(np.allclose(typed_array, base_array.astype(typecode)))
            typed_array = None
            typed_array = StorageNumpy(None, storage_id, tablename)
            self.assertTrue(np.allclose(typed_array, base_array.astype(typecode)))

        for typecode in np.typecodes['UnsignedInteger']:
            if typecode == 'P':
                # TODO For now skip arrays made of pointers
                pass
            storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, tablename + typecode)
            typed_array = StorageNumpy(base_array.astype(typecode), storage_id, tablename)
            self.assertTrue(np.allclose(typed_array, base_array.astype(typecode)))
            typed_array = None
            typed_array = StorageNumpy(None, storage_id, tablename)
            self.assertTrue(np.allclose(typed_array, base_array.astype(typecode)))

if __name__ == '__main__':
    unittest.main()
