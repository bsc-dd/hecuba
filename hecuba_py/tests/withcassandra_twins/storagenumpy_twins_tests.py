import gc
import unittest

from hecuba import config, StorageNumpy
import uuid
import numpy as np

from storage.api import getByID


class StorageNumpyTwinsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = config.execution_name
        config.execution_name = "StorageNumpyTest".lower()

    @classmethod
    def tearDownClass(cls):
#        config.session.execute("DROP KEYSPACE IF EXISTS {}".format(config.execution_name))
#        config.session.execute("DROP KEYSPACE IF EXISTS {}_arrow".format(config.execution_name))
        config.execution_name = cls.old

    # Create a new keyspace per test
    def setUp(self):
        self.ksp = config.execution_name

    def tearDown(self):
        pass

    table = 'numpy_twin_test'

    def test_twin_volatile_from_numpy(self):

        n = np.arange(3*4).reshape(3,4)

        s = StorageNumpy(n)

        self.assertTrue(s._twin_ref is not None)
        self.assertEqual(s._twin_id, None)
        self.assertEqual(s._twin_name, None)
        self.assertEqual(s._twin_ref._name, None)
        self.assertEqual(s._twin_ref.storage_id, None)
        self.assertEqual(n.T.shape, s._twin_ref.shape)
        self.assertTrue(np.allclose(s._twin_ref, n.T))

    def test_twin_volatile_from_storagenumpy(self):

        n = np.arange(3*4).reshape(3,4)

        s = StorageNumpy(n)

        s2 = StorageNumpy(s)

        self.assertTrue(s2._twin_ref is not None)
        self.assertEqual(s2._twin_id, None)
        self.assertEqual(s2._twin_name, None)
        self.assertEqual(s2._twin_ref._name, None)
        self.assertEqual(s2._twin_ref.storage_id, None)
        self.assertEqual(n.T.shape, s2._twin_ref.shape)
        self.assertTrue(np.array_equal(s2._twin_ref, n.T))

    def test_twin_persistent(self):

        n = np.arange(3*4).reshape(3,4)

        s = StorageNumpy(n, 'persistent')

        self.assertTrue(s._twin_id is not None)
        self.assertEqual(s._twin_name, self.ksp+'_arrow.persistent_arrow')
        self.assertTrue(np.array_equal(s._twin_ref, n.T))
        self.assertEqual(s._twin_ref._name, self.ksp+'_arrow.persistent_arrow')
        self.assertEqual(s._twin_ref.storage_id, s._twin_id)
        self.assertEqual(n.T.shape, s._twin_ref.shape)
        self.assertEqual(s._build_args.twin_id, s._twin_id) #stored data in cassandra
        res = config.session.execute(
                        "SELECT twin_id FROM hecuba.istorage WHERE storage_id = %s",
                        [s.storage_id] )
        self.assertEqual(res.one().twin_id, s._twin_id)

    def test_twin_persistent_manual(self):
        n = np.arange(3*4).reshape(3,4)

        s = StorageNumpy(n)
        s.make_persistent('manual_pers')

        self.assertTrue(s._twin_id is not None)
        self.assertEqual(s._twin_name, self.ksp+'_arrow.manual_pers_arrow')
        self.assertEqual(s._twin_ref._name, self.ksp+'_arrow.manual_pers_arrow')
        self.assertEqual(s._twin_ref.storage_id, s._twin_id)
        self.assertEqual(n.T.shape, s._twin_ref.shape)
        self.assertEqual(s._build_args.twin_id, s._twin_id) #stored data in cassandra
        res = config.session.execute(
                        "SELECT twin_id FROM hecuba.istorage WHERE storage_id = %s",
                        [s.storage_id] )
        self.assertEqual(res.one().twin_id, s._twin_id)
        self.assertTrue(np.allclose(s._twin_ref, n.T))

    def test_twin_persistent_from_storagenumpy(self):

        n = np.arange(3*4).reshape(3,4)

        s = StorageNumpy(n, 'pers_from_sn')

        s2 = StorageNumpy(s)    # Create a volatile SN

        self.assertTrue(s2._twin_ref is not None)
        self.assertEqual(s2._twin_id, None)
        self.assertEqual(s2._twin_name, None)
        self.assertEqual(s2._twin_ref._name, None)
        self.assertEqual(s2._twin_ref.storage_id, None)
        self.assertEqual(n.T.shape, s2._twin_ref.shape)
        self.assertTrue(np.allclose(s2._twin_ref, n.T))

# FIXME currently this case is not implemented
#    def test_twin_persistent_from_storagenumpy2(self):
#
#        n = np.arange(3*4).reshape(3,4)
#
#        s = StorageNumpy(n, 'kk')
#
#        s2 = StorageNumpy(s, 'ooops') #The name should be ignored
#
#        self.assertTrue(s2._twin_id is not None)
#        self.assertEqual(s2._twin_name, self.ksp+'.harrow_kk')
#        self.assertEqual(s2._twin_ref._name, self.ksp+'.harrow_kk')
#        self.assertEqual(s2._twin_ref.storage_id, s2._twin_id)
#        self.assertEqual(s2._build_args.twin_id, s2._twin_id) #stored data in cassandra
#        self.assertEqual(n.T.shape, s2._twin_ref.shape)
#        self.assertTrue(np.array_equal(s2._twin_ref, n.T))

    def test_load_persistent_twin_by_name(self):

        n = np.arange(3*4).reshape(3,4)

        s = StorageNumpy(n, 'load_by_name')
        sid = s.storage_id

        del s

        s2 = StorageNumpy(None, 'load_by_name')

        self.assertTrue(s2._twin_id is not None)
        self.assertEqual(s2._twin_name, self.ksp+'_arrow.load_by_name_arrow')
        self.assertEqual(s2._twin_ref._name, self.ksp+'_arrow.load_by_name_arrow')
        self.assertEqual(s2._twin_ref.storage_id, s2._twin_id)
        self.assertEqual(sid, s2.storage_id)
        self.assertEqual(n.T.shape, s2._twin_ref.shape)
        self.assertTrue(np.allclose(s2._twin_ref, n.T))
        self.assertTrue(np.allclose(s2, n))

    def test_load_persistent_twin_by_id(self):

        n = np.arange(3*4).reshape(3,4)

        s = StorageNumpy(n, 'load_by_id')
        sid = s.storage_id

        del s

        s2 = StorageNumpy(None, None, sid)

        self.assertTrue(s2._twin_ref is not None)
        self.assertTrue(s2._twin_id is not None)
        self.assertEqual(s2._twin_name, self.ksp+'_arrow.load_by_id_arrow')
        self.assertEqual(s2._twin_ref._name, self.ksp+'_arrow.load_by_id_arrow')
        self.assertEqual(s2._twin_ref.storage_id, s2._twin_id)
        self.assertEqual(sid, s2.storage_id)
        self.assertEqual(n.T.shape, s2._twin_ref.shape)
        self.assertTrue(np.allclose(s2, n))
        self.assertTrue(np.allclose(s2._twin_ref, n.T))

    def test_load_persistent_twin_by_name_and_id(self):

        n = np.arange(3*4).reshape(3,4)

        s = StorageNumpy(n, 'by_name_and_id')
        sid = s.storage_id

        del s

        s2 = StorageNumpy(None, 'by_name_and_id', sid)

        #FIXME
        #self.assertTrue(s2._twin_id is not None)
        #self.assertEqual(s2._twin_name, self.ksp+'.harrow_kk')
        #self.assertEqual(s2._twin_ref._name, self.ksp+'.harrow_kk')
        #self.assertEqual(s2._twin_ref.storage_id, s2._twin_id)
        #self.assertEqual(sid, s2.storage_id)
        #self.assertTrue(np.array_equal(s2._twin_ref, n.T))
        pass

if __name__ == '__main__':
    unittest.main()
