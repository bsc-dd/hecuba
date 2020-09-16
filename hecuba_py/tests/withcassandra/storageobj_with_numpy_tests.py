import unittest

import numpy as np
from hecuba import config, StorageObj


class TestStorageObjNumpy(StorageObj):
    '''
       @ClassField mynumpy numpy.ndarray
    '''
    pass


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
        #config.session.execute("DROP KEYSPACE IF EXISTS {}".format(self.ksp))
        pass

    table = 'numpy_test'

    def test_numpy_reserved_all_cluster_2D(self):
        num = 0
        no = TestStorageObjNumpy(self.ksp+".numpy_test_%d" % num)
        no.mynumpy = np.arange(40000).reshape((200, 200))
        myobj2 = TestStorageObjNumpy(self.ksp+".numpy_test_%d" % num)
        chunk = myobj2.mynumpy[slice(None, None, None)]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(40000).reshape((200, 200))
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_reserved_2d_1cluster(self):
        coordinates = (slice(0, 50, None), slice(0, 50, None))
        num = 1
        no = TestStorageObjNumpy(self.ksp+".numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy(self.ksp+".numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100, 100))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_reserved_2d_2cluster(self):
        coordinates = (slice(0, 22, None), slice(0, 44, None))
        num = 2
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100, 100))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_reserved_2d_3cluster(self):
        coordinates = (slice(0, 45, None), slice(0, 22, None))
        num = 3
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100, 100))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_reserved_2d_4cluster(self):
        coordinates = (slice(0, 45, None), slice(0, 45, None))
        num = 4
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100, 100))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_reserved_3d_1cluster(self):
        coordinates = (slice(0, 8, None), slice(0, 8, None), slice(0, 8, None))
        num = 5
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(1000).reshape((10, 10, 10))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(1000).reshape((10, 10, 10))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_2D_slice_right(self):
        num = 5
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[slice(None, 5, None), slice(None, 5, None)]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100, 100))
        test_numpy = test_numpy[slice(None, 5, None), slice(None, 5, None)]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_2D_slice_left(self):
        num = 6
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[slice(5, None, None), slice(5, None, None)]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100, 100))
        test_numpy = test_numpy[slice(5, None, None), slice(5, None, None)]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_1D_slice_right(self):
        num = 5
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000)
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[slice(None, 5, None)]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000)
        test_numpy = test_numpy[slice(None, 5, None)]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_1D_slice_left(self):
        num = 5
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000)
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[slice(5, None, None)]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000)
        test_numpy = test_numpy[slice(5, None, None)]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_1D_none_slices(self):
        num = 5
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000)
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[slice(None, None, None)]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000)
        test_numpy = test_numpy[slice(None, None, None)]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_2D_some_none_slices(self):
        num = 30
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100,100))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[slice(None, None, None), slice(4, 100, None)]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100,100))
        test_numpy = test_numpy[slice(None, None, None), slice(4, 100, None)]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_1D_none_slice_not_a_list(self):
        num = 25
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000)
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[slice(None, None, None)]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000)
        test_numpy = test_numpy[slice(None, None, None)]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_1D_list_1_slice(self):
        num = 26
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000)
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[slice(2, 20, None)]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000)
        test_numpy = test_numpy[slice(2, 20, None)]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_reserved_4d_1cluster(self):
        coordinates = (slice(0, 8, None), slice(0, 8, None), slice(0, 8, None), slice(0, 8, None))
        num = 6
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((10, 10, 10, 10))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((10, 10, 10, 10))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_reserved_5d_1cluster(self):
        coordinates = (slice(0, 5, None), slice(0, 5, None), slice(0, 5, None), slice(0, 5, None), slice(0, 5, None))
        num = 13
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(100000).reshape((10, 10, 10, 10, 10))
        import gc
        del no
        gc.collect()
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(100000).reshape((10, 10, 10, 10, 10))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_all_slices_out_of_bounds(self):
        coordinates = (slice(200, 1000, None), slice(200, 1000, None))
        num = 14
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100, 100))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_all_slices_out_of_bounds2(self):
        coordinates = (slice(20, 100, None), slice(20, 100, None), slice(20, 100, None))
        num = 15
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(1000).reshape((10, 10, 10))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(1000).reshape((10, 10, 10))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_some_slices_out_of_bounds(self):
        coordinates = (slice(50, 150, None), slice(50, 150, None))
        num = 16
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(10000).reshape((100, 100))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_numpy_some_slices_out_of_bounds2(self):
        coordinates = (slice(50, 150, None), slice(50, 150, None), slice(5, 150, None))
        num = 16
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(1000).reshape((10, 10, 10))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        result = chunk.view(np.ndarray)
        test_numpy = np.arange(1000).reshape((10, 10, 10))
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(result, test_numpy))

    def test_write_1value_all_numpy(self):
        coordinates = (slice(0, 45, None), slice(0, 45, None))
        num = 7
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        chunk[:] = 1
        test_numpy = np.arange(10000).reshape((100, 100))[coordinates[0], coordinates[1]]
        test_numpy[:] = 1
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))

    def test_write_1value_all_numpy_2coord(self):
        coordinates = (slice(0, 45, None), slice(0, 45, None))
        num = 8
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        set_coord = (slice(0, 10, None), slice(0, 10, None))
        chunk[set_coord] = 1
        test_numpy = np.arange(10000).reshape((100, 100))[coordinates[0], coordinates[1]]
        test_numpy[set_coord] = 1
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))

    def test_write_nvalue_all_numpy_2coord(self):
        coordinates = (slice(0, 45, None), slice(0, 45, None))
        num = 9
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        set_coord = (slice(0, 10, None), slice(0, 10, None))
        chunk[set_coord] = np.arange(100).reshape((10, 10))
        test_numpy = np.arange(10000).reshape((100, 100))[coordinates[0], coordinates[1]]
        test_numpy[set_coord] = np.arange(100).reshape((10, 10))
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))

    def test_write_nvalue2_all_numpy_2coord_rww(self):
        coordinates = (slice(0, 45, None), slice(0, 45, None))
        num = 10
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        set_coord = (slice(0, 10, None), slice(0, 10, None))
        chunk[set_coord] = np.arange(100).reshape((10, 10))
        chunk[slice(10, 20, None), slice(10, 20, None)] = np.arange(100).reshape((10, 10))
        test_numpy = np.arange(10000).reshape((100, 100))[coordinates[0], coordinates[1]]
        test_numpy[set_coord] = np.arange(100).reshape((10, 10))
        test_numpy[slice(10, 20, None), slice(10, 20, None)] = np.arange(100).reshape((10, 10))
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))

    def test_write_nvalue2_all_numpy_2coord_rwr(self):
        coordinates = (slice(0, 45, None), slice(0, 45, None))
        num = 11
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.arange(10000).reshape((100, 100))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy[coordinates]
        set_coord = (slice(0, 10, None), slice(0, 10, None))
        chunk[set_coord] = np.arange(100).reshape((10, 10))
        test_numpy = np.arange(10000).reshape((100, 100))[coordinates[0], coordinates[1]]
        test_numpy[set_coord] = np.arange(100).reshape((10, 10))
        coordinates = (slice(0, 20, None), slice(0, 20, None))
        chunk = chunk[coordinates]
        test_numpy = test_numpy[coordinates]
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))

    def test_write_nvalue2_all_numpy_2coord_w(self):
        coordinates = (slice(30, 45, None), slice(30, 45, None))
        num = 12
        no = TestStorageObjNumpy("numpy_test_%d" % num)
        no.mynumpy = np.zeros((100, 100))
        myobj2 = TestStorageObjNumpy("numpy_test_%d" % num)
        chunk = myobj2.mynumpy
        chunk[coordinates] = 1
        test_numpy = np.zeros((100, 100))
        test_numpy[coordinates] = 1
        self.assertTrue(np.allclose(chunk.view(np.ndarray), test_numpy))

if __name__ == '__main__':
    unittest.main()
