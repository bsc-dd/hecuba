import unittest

from storage.api import getByID
from hecuba.hdict import StorageDict


class ApiTestSDict(StorageDict):
    '''
    @TypeSpec <<key:int>, value:double>
    '''

class StorageApi_Tests(unittest.TestCase):
    def setUp(self):
        pass

    def class_type_test(self):
        base_dict = ApiTestSDict('test.api_sdict')
        storage_id = base_dict.getID()
        del base_dict

        rebuild_dict = getByID(storage_id)
        self.assertTrue(isinstance(rebuild_dict, ApiTestSDict))