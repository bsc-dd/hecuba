import unittest

from hecuba import config, StorageDict
from hecuba.IStorage import IStorage


class PersistentDict(StorageDict):
    '''
    @TypeSpec dict<<key:int>, value:double>
    '''


class IStorageTests(unittest.TestCase):
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
        self.current_ksp = "IStorageTests{}".format(config.NUM_TEST).lower()
        config.execution_name = self.current_ksp

    def tearDown(self):
        #config.session.execute("DROP KEYSPACE IF EXISTS {}".format(self.current_ksp))
        pass

    def stop_persistent_method_test(self):
        key = 123
        value = 456
        name = 'istorage_pers'

        base_dict = PersistentDict()

        def check_stop_pers():
            assert (isinstance(base_dict, IStorage))
            base_dict.stop_persistent()

        self.assertRaises(RuntimeError, check_stop_pers)
        # PyCOMPSs requires uuid of type str

        base_dict.make_persistent(name)

        base_dict[key] = value

        base_dict.stop_persistent()

        self.assertIsNone(base_dict.storage_id)
        self.assertIsNone(base_dict.storage_id)

        self.assertRaises(RuntimeError, check_stop_pers)

        base_dict.make_persistent(name)

        self.assertEqual(base_dict[key], value)

        base_dict.stop_persistent()

        self.assertRaises(RuntimeError, check_stop_pers)

        external_dict = PersistentDict(name)
        self.assertEqual(external_dict[key], value)

    def delete_persistent_method_test(self):
        key = 123
        value = 456
        name = 'istorage_pers'

        base_dict = PersistentDict()

        def check_stop_pers():
            assert (isinstance(base_dict, IStorage))
            base_dict.stop_persistent()

        def check_del_pers():
            assert (isinstance(base_dict, IStorage))
            base_dict.delete_persistent()

        self.assertRaises(RuntimeError, check_del_pers)
        # PyCOMPSs requires uuid of type str

        base_dict.make_persistent(name)

        base_dict[key] = value

        base_dict.delete_persistent()

        self.assertIsNone(base_dict.storage_id)
        self.assertIsNone(base_dict.storage_id)

        self.assertRaises(RuntimeError, check_del_pers)
        self.assertRaises(RuntimeError, check_stop_pers)

        base_dict.make_persistent(name)

        def get_key():
            res = base_dict[key]

        self.assertRaises(KeyError, get_key)

        base_dict.delete_persistent()

        self.assertRaises(RuntimeError, check_del_pers)
        self.assertRaises(RuntimeError, check_stop_pers)

        external_dict = PersistentDict(name)

        def get_key_ext():
            res = external_dict[key]

        self.assertRaises(KeyError, get_key_ext)


if __name__ == '__main__':
    unittest.main()
