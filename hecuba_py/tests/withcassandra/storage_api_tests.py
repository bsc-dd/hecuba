import unittest

from storage.api import getByID
from ..app.words import Words
from hecuba import StorageDict, StorageNumpy
from hecuba import StorageObj as StorageObject
from hecuba import config


class ApiTestSDict(StorageDict):
    '''
    @TypeSpec dict<<key:int>, value:double>
    '''


class ApiTestSObject(StorageObject):
    '''
    @ClassField attr int
    @ClassField attr2 str
    '''


class StorageApi_Tests(unittest.TestCase):
    def class_type_test(self):
        base_dict = ApiTestSDict('test.api_sdict')
        # PyCOMPSs requires uuid of type str
        storage_id = str(base_dict.storage_id)
        del base_dict

        rebuild_dict = getByID(storage_id)
        self.assertTrue(isinstance(rebuild_dict, ApiTestSDict))
        rebuild_dict.delete_persistent()
        config.session.execute( "DROP TABLE IF  EXISTS test.api_sdict;")

    def object_id_is_str_test(self):
        memory_obj = ApiTestSDict()

        self.assertTrue(hasattr(memory_obj, 'getID'))
        self.assertIsInstance(memory_obj.getID(), str, "PyCOMPSs specs states that getID should return a string")

        pers_dict = ApiTestSDict('test.api_id_str')
        self.assertTrue(hasattr(pers_dict, 'getID'))
        self.assertIsInstance(pers_dict.getID(), str, "PyCOMPSs specs states that getID should return a string")
        pers_dict.delete_persistent()
        config.session.execute( "DROP TABLE IF  EXISTS test.api_id_str;")

    def test_getByID_block(self):
        # ki = KeyIter('testspace', 'tt', 'app.words.Words', 'fake-id', ['position'])
        SO = Words('so')
        b = next(SO.split())
        new_block = getByID(b.storage_id)
        self.assertEqual(b.storage_id, new_block.storage_id)
        self.assertEqual(b, new_block)
        SO.delete_persistent()
        config.session.execute("DROP TABLE IF EXISTS my_app.so")
        config.session.execute("DROP TABLE IF EXISTS my_app.so_0")
        config.session.execute("DROP TABLE IF EXISTS my_app.so_1")

    def test_getByID_storage_obj(self):
        b = Words('testspace.tt')
        new_block = getByID(b.storage_id)
        self.assertEqual(b, new_block)
        b.delete_persistent()
        config.session.execute("DROP TABLE IF EXISTS testspace.tt")

    def test_get_by_alias(self):
        attr = 123
        attr2 = "textattribute"
        obj = ApiTestSObject('test.api_by_alias')
        obj.attr = attr
        obj.attr2 = attr2
        del obj
        rebuild = ApiTestSObject.get_by_alias('test.api_by_alias')
        self.assertEqual(rebuild.attr, attr)
        self.assertEqual(rebuild.attr2, attr2)

        obj2 = ApiTestSDict('test.api_by_alias2')
        rebuild = ApiTestSDict.get_by_alias('test.api_by_alias2')
        self.assertEqual(obj2, rebuild)
        #clean up
        rebuild.delete_persistent()
        rebuild = ApiTestSObject.get_by_alias('test.api_by_alias')
        rebuild.delete_persistent()
        config.session.execute("DROP TABLE IF EXISTS test.api_by_alias")
        config.session.execute("DROP TABLE IF EXISTS test.api_by_alias2")
