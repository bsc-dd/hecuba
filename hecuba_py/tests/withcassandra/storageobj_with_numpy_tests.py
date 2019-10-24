import unittest

import numpy as np
from hecuba import config, StorageObj


class TestStorageObjNumpy(StorageObj):
    '''
       @ClassField mynumpy numpy.ndarray
    '''
    pass


class StorageNumpyTest(unittest.TestCase):
    table = 'numpy_test'
    ksp = 'my_app'

    def test_numpy_reserved_all_cluster_2D(self):
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_0;")
        coordinates = slice(None, None, None)
        num = 0
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(40000).reshape((200, 200))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(40000).reshape((200, 200))
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_reserved_2d_1cluster(self):
        coordinates = [slice(0, 50, None), slice(0, 50, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_1;")
        num = 1
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100, 100))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_reserved_2d_2cluster(self):
        coordinates = [slice(0, 22, None), slice(0, 44, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_2;")
        num = 2
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100, 100))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_reserved_2d_3cluster(self):
        coordinates = [slice(0, 45, None), slice(0, 22, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_3;")
        num = 3
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100, 100))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_reserved_2d_4cluster(self):
        coordinates = [slice(0, 45, None), slice(0, 45, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_4;")
        num = 4
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100, 100))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_reserved_3d_1cluster(self):
        coordinates = [slice(0, 8, None), slice(0, 8, None), slice(0, 8, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_5;")
        num = 5
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(1000).reshape((10, 10, 10))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(1000).reshape((10, 10, 10))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_1_slice(self):
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_5;")
        num = 5
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100,100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[:25]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100,100))
        test_numpy = test_numpy[:25]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_reserved_4d_1cluster(self):
        coordinates = [slice(0, 8, None), slice(0, 8, None), slice(0, 8, None), slice(0, 8, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_6;")
        num = 6
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((10, 10, 10, 10))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((10, 10, 10, 10))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_reserved_5d_1cluster(self):
        coordinates = [slice(0, 5, None), slice(0, 5, None), slice(0, 5, None), slice(0, 5, None), slice(0, 5, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_13;")
        num = 13
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(100000).reshape((10, 10, 10, 10, 10))
        import gc
        del no
        gc.collect()
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(100000).reshape((10, 10, 10, 10, 10))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_all_slices_out_of_bounds(self):
        coordinates = [slice(200, 1000, None), slice(200, 1000, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_1;")
        num = 14
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100, 100))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_all_slices_out_of_bounds2(self):
        coordinates = [slice(20, 100, None), slice(20, 100, None), slice(20, 100, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_1;")
        num = 15
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(1000).reshape((10, 10, 10))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(1000).reshape((10, 10, 10))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_some_slices_out_of_bounds(self):
        coordinates = [slice(50, 150, None), slice(50, 150, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_1;")
        num = 16
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100, 100))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_some_slices_out_of_bounds2(self):
        coordinates = [slice(50, 150, None), slice(50, 150, None), slice(5, 150, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_1;")
        num = 16
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(1000).reshape((10, 10, 10))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(1000).reshape((10, 10, 10))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_write_1value_all_numpy(self):
        coordinates = [slice(0, 45, None), slice(0, 45, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_7;")
        num = 7
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        chunk[:] = 1
        test_numpy = np.arange(10000).reshape((100, 100))[coordinates[0], coordinates[1]]
        test_numpy[:] = 1
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))

    def test_write_1value_all_numpy_2coord(self):
        coordinates = [slice(0, 45, None), slice(0, 45, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_8;")
        num = 8
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        set_coord = [slice(0, 10, None), slice(0, 10, None)]
        chunk[set_coord] = 1
        test_numpy = np.arange(10000).reshape((100, 100))[coordinates[0], coordinates[1]]
        test_numpy[set_coord] = 1
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))

    def test_write_nvalue_all_numpy_2coord(self):
        coordinates = [slice(0, 45, None), slice(0, 45, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_9;")
        num = 9
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        set_coord = [slice(0, 10, None), slice(0, 10, None)]
        chunk[set_coord] = np.arange(100).reshape((10, 10))
        test_numpy = np.arange(10000).reshape((100, 100))[coordinates[0], coordinates[1]]
        test_numpy[set_coord] = np.arange(100).reshape((10, 10))
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))

    def test_write_nvalue2_all_numpy_2coord_rww(self):
        coordinates = [slice(0, 45, None), slice(0, 45, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_10;")
        num = 10
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        set_coord = [slice(0, 10, None), slice(0, 10, None)]
        chunk[set_coord] = np.arange(100).reshape((10, 10))
        chunk[slice(10, 20, None), slice(10, 20, None)] = np.arange(100).reshape((10, 10))
        test_numpy = np.arange(10000).reshape((100, 100))[coordinates[0], coordinates[1]]
        test_numpy[set_coord] = np.arange(100).reshape((10, 10))
        test_numpy[slice(10, 20, None), slice(10, 20, None)] = np.arange(100).reshape((10, 10))
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))

    def test_write_nvalue2_all_numpy_2coord_rwr(self):
        coordinates = [slice(0, 45, None), slice(0, 45, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_11;")
        num = 11
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        set_coord = [slice(0, 10, None), slice(0, 10, None)]
        chunk[set_coord] = np.arange(100).reshape((10, 10))
        test_numpy = np.arange(10000).reshape((100, 100))[coordinates[0], coordinates[1]]
        test_numpy[set_coord] = np.arange(100).reshape((10, 10))
        coordinates = [slice(0, 20, None), slice(0, 20, None)]
        chunk = chunk[coordinates]
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))

    def test_write_nvalue2_all_numpy_2coord_w(self):
        coordinates = [slice(30, 45, None), slice(30, 45, None)]
        config.session.execute("DROP TABLE IF EXISTS myapp.numpy_test_12;")
        num = 12
        no = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        no.mynumpy = np.zeros((100, 100))
        myobj2 = TestStorageObjNumpy("my_app.numpy_test_%d" % num)
        chunk = myobj2.mynumpy
        chunk[coordinates] = 1
        test_numpy = np.zeros((100, 100))
        test_numpy[coordinates] = 1
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))


if __name__ == '__main__':
    unittest.main()
