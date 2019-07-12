import unittest

from hecuba import config, StorageDict
from hecuba.IStorage import IStorage


class PersistentDict(StorageDict):
    '''
    @TypeSpec dict<<key:int>, value:double>
    '''


class IStorageTests(unittest.TestCase):
    def stop_persistent_method_test(self):
        from hecuba.tools import storage_id_from_name
        config.session.execute(
            "DROP TABLE IF  EXISTS test.istorage_pers;")
        config.session.execute(
            "DELETE FROM hecuba.istorage WHERE storage_id = {}".format(storage_id_from_name("test.istorage_pers")))

        key = 123
        value = 456
        name = 'test.istorage_pers'

        base_dict = PersistentDict()

        def check_stop_pers():
            assert (isinstance(base_dict, IStorage))
            base_dict.stop_persistent()

        self.assertRaises(AttributeError, check_stop_pers)
        # PyCOMPSs requires uuid of type str

        base_dict.make_persistent(name)

        base_dict[key] = value

        base_dict.stop_persistent()

        self.assertIsNone(base_dict.storage_id)
        self.assertIsNone(base_dict.storage_id)

        self.assertRaises(AttributeError, check_stop_pers)

        base_dict.make_persistent(name)

        self.assertEqual(base_dict[key], value)

        base_dict.stop_persistent()

        self.assertRaises(AttributeError, check_stop_pers)

        external_dict = PersistentDict(name)
        self.assertEqual(external_dict[key], value)

    def delete_persistent_method_test(self):
        from hecuba.tools import storage_id_from_name
        config.session.execute(
            "DROP TABLE IF  EXISTS test.istorage_pers;")
        config.session.execute(
            "DELETE FROM hecuba.istorage WHERE storage_id = {}".format(storage_id_from_name("test.istorage_pers")))

        key = 123
        value = 456
        name = 'test.istorage_pers'

        base_dict = PersistentDict()

        def check_stop_pers():
            assert (isinstance(base_dict, IStorage))
            base_dict.stop_persistent()

        def check_del_pers():
            assert (isinstance(base_dict, IStorage))
            base_dict.delete_persistent()

        self.assertRaises(AttributeError, check_del_pers)
        # PyCOMPSs requires uuid of type str

        base_dict.make_persistent(name)

        base_dict[key] = value

        base_dict.delete_persistent()

        self.assertIsNone(base_dict.storage_id)
        self.assertIsNone(base_dict.storage_id)

        self.assertRaises(AttributeError, check_del_pers)
        self.assertRaises(AttributeError, check_stop_pers)

        base_dict.make_persistent(name)

        def get_key():
            res = base_dict[key]

        self.assertRaises(KeyError, get_key)

        base_dict.delete_persistent()

        self.assertRaises(AttributeError, check_del_pers)
        self.assertRaises(AttributeError, check_stop_pers)

        external_dict = PersistentDict(name)

        def get_key_ext():
            res = external_dict[key]

        self.assertRaises(KeyError, get_key_ext)


if __name__ == '__main__':
    unittest.main()
