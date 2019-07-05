import unittest

from storage.api import getByID
from ..app.words import Words
from hecuba import config, StorageDict


class ApiTestSDict(StorageDict):
    '''
    @TypeSpec dict<<key:int>, value:double>
    '''


class StorageApi_Tests(unittest.TestCase):
    def class_type_test(self):
        base_dict = ApiTestSDict('test.api_sdict')
        # PyCOMPSs requires uuid of type str
        storage_id = str(base_dict.get_id())
        del base_dict

        rebuild_dict = getByID(storage_id)
        self.assertTrue(isinstance(rebuild_dict, ApiTestSDict))

    def object_id_is_str_test(self):
        memory_obj = ApiTestSDict()

        self.assertTrue(hasattr(memory_obj, 'getID'))
        self.assertIsInstance(memory_obj.getID(), str, "PyCOMPSs specs states that getID should return a string")

        pers_dict = ApiTestSDict('test.api_id_str')
        self.assertTrue(hasattr(pers_dict, 'getID'))
        self.assertIsInstance(pers_dict.getID(), str, "PyCOMPSs specs states that getID should return a string")

    def test_getByID_block(self):
        # ki = KeyIter('testspace', 'tt', 'app.words.Words', 'fake-id', ['position'])
        from hecuba import config
        config.session.execute("DROP TABLE IF EXISTS my_app.so")
        config.session.execute("DROP TABLE IF EXISTS my_app.so_0")
        config.session.execute("DROP TABLE IF EXISTS my_app.so_1")
        SO = Words('so')
        b = next(SO.split())
        new_block = getByID(b.get_id())
        self.assertEqual(b.get_id(), new_block.get_id())
        self.assertEqual(b, new_block)

    def test_getByID_storage_obj(self):
        b = Words('testspace.tt')
        new_block = getByID(b.get_id())
        self.assertEqual(b, new_block)
