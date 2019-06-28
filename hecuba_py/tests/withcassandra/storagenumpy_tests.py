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

    def test_numpy_reserved_all_cluster(self):
        coordinates = [slice(0,60,None), slice(0,60,None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_10000;")
        size = 10000
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        no.mynumpy = np.arange(40000).reshape((200, 200))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        chunk = myobj2.mynumpy[coordinates] # we are getting the first cluster
        self.assertEqual(chunk[87,87],17487)
        self.assertNotEqual(chunk[87,88], 17488)


    def test_numpy_reserved_1cluster(self):
        coordinates = [slice(0,40,None), slice(0,40,None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_10000;")
        size = 10000
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        chunk = myobj2.mynumpy[coordinates] # we are getting the first cluster
        self.assertEqual(chunk[0,43],43)
        self.assertEqual(chunk[43,43], 4343)
        self.assertNotEqual(chunk[0, 44], 4400)
        self.assertNotEqual(chunk[44, 44], 4444)
        self.assertNotEqual(chunk[45, 45], 4544)

    def test_numpy_reserved_2cluster(self):
        coordinates = [slice(0,22,None), slice(0,44,None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_10000;")
        size = 10000
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        chunk = myobj2.mynumpy[coordinates] # we are getting the first cluster
        self.assertEqual(chunk[0,44],44)
        self.assertEqual(chunk[43,87], 4387)
        self.assertNotEqual(chunk[43, 0], 43)
        self.assertNotEqual(chunk[43, 88], 4388)

    def test_numpy_reserved_3cluster(self):
        coordinates = [slice(0,45,None), slice(0,22,None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_10000;")
        size = 10000
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        chunk = myobj2.mynumpy[coordinates] # we are getting the first cluster
        self.assertEqual(chunk[44, 0], 4400)
        self.assertEqual(chunk[87,43],8743)
        self.assertNotEqual(chunk[87, 44], 8744)
        self.assertNotEqual(chunk[88, 44], 8844)


    def test_numpy_reserved_4cluster(self):
        coordinates = [slice(0,45,None), slice(0,45,None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_10000;")
        size = 10000
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % size)
        chunk = myobj2.mynumpy[coordinates] # we are getting the first cluster
        self.assertNotEqual(chunk[43, 43], 4443)
        self.assertEqual(chunk[44,44],4444)
        self.assertEqual(chunk[87,87], 8787)
        self.assertNotEqual(chunk[88, 88], 8788)


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
