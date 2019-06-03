import unittest

from storage.api import getByID
from hecuba import config, StorageDict


class ApiTestSDict(StorageDict):
    '''
    @TypeSpec dict<<key:int>, value:double>
    '''


class StorageApi_Tests(unittest.TestCase):
    def class_type_test(self):
        base_dict = ApiTestSDict('test.api_sdict')
        # PyCOMPSs requires uuid of type str
        storage_id = str(base_dict.getID())
        del base_dict

        rebuild_dict = getByID(storage_id)
        self.assertTrue(isinstance(rebuild_dict, ApiTestSDict))
