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
        self.current_ksp = "StorageApi_Tests{}".format(config.NUM_TEST).lower()
        config.execution_name = self.current_ksp

    def tearDown(self):
        #config.session.execute("DROP KEYSPACE IF EXISTS {}".format(self.current_ksp))
        pass

    def class_type_test(self):
        base_dict = ApiTestSDict('api_sdict')
        # PyCOMPSs requires uuid of type str
        storage_id = str(base_dict.storage_id)
        del base_dict

        rebuild_dict = getByID(storage_id)
        self.assertTrue(isinstance(rebuild_dict, ApiTestSDict))
        rebuild_dict.delete_persistent()

    def object_id_is_str_test(self):
        memory_obj = ApiTestSDict()

        self.assertTrue(hasattr(memory_obj, 'getID'))
        self.assertIsInstance(memory_obj.getID(), str, "PyCOMPSs specs states that getID should return a string")

        pers_dict = ApiTestSDict('api_id_str')
        self.assertTrue(hasattr(pers_dict, 'getID'))
        self.assertIsInstance(pers_dict.getID(), str, "PyCOMPSs specs states that getID should return a string")
        pers_dict.delete_persistent()

    #DISABLED until split for storageobj works
    #def test_getByID_block(self):
    #    # ki = KeyIter('testspace', 'tt', 'app.words.Words', 'fake-id', ['position'])
    #    SO = Words('so')
    #    b = next(SO.split())
    #    new_block = getByID(b.storage_id)
    #    self.assertEqual(b.storage_id, new_block.storage_id)
    #    self.assertEqual(b, new_block)
    #    SO.delete_persistent()

    def test_getByID_storage_obj(self):
        b = Words('tt')
        new_block = getByID(b.storage_id)
        self.assertEqual(b, new_block)
        b.delete_persistent()

    def test_get_by_alias(self):
        attr = 123
        attr2 = "textattribute"
        obj = ApiTestSObject('api_by_alias')
        obj.attr = attr
        obj.attr2 = attr2
        del obj
        rebuild = ApiTestSObject.get_by_alias('api_by_alias')
        self.assertEqual(rebuild.attr, attr)
        self.assertEqual(rebuild.attr2, attr2)

        obj2 = ApiTestSDict('api_by_alias2')
        rebuild = ApiTestSDict.get_by_alias('api_by_alias2')
        self.assertEqual(obj2, rebuild)
        #clean up
        rebuild.delete_persistent()
        rebuild = ApiTestSObject.get_by_alias('api_by_alias')
        rebuild.delete_persistent()
