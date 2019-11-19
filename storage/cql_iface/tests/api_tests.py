import unittest
import uuid

import numpy

from storage.cql_iface.cql_iface import CQLIface
from storage.cql_iface.tests.mockIStorage import IStorage
from storage.cql_iface.tests.mockStorageObj import StorageObj
from storage.cql_iface.tests.mockhdict import StorageDict
from storage.cql_iface.tests.mockhnumpy import StorageNumpy


class TestClass(IStorage):

    def __new__(cls, name='', *args, **kwargs):
        toret = super(TestClass, cls).__new__(cls)
        return toret

    def __init__(self, *args, **kwargs):
        super(TestClass, self).__init__()


class mockClass(IStorage):
    pass


class HfetchTests(unittest.TestCase):

    def test_instantiate(self):
        result = CQLIface()
        self.assertIsNotNone(result)

    def test_add_data_model_except_not_dict_type(self):
        with self.assertRaises(TypeError):
            data_model = "a"
            storage = CQLIface()
            # Register data models
            storage.check_definition(data_model)

    def test_add_data_model_except_invalid_class(self):
        with self.assertRaises(TypeError):
            data_model = {"type": "a", "value_id": {"k": str}, "fields": {"a": str}}
            storage = CQLIface()
            # Register data models
            storage.check_definition(data_model)

    def test_add_data_model_except_invalid_format(self):
        with self.assertRaises(KeyError):
            data_model = {"type": mockClass, "": {"k": str}, "fields": {"a": str}}
            storage = CQLIface()
            # Register data models
            storage.check_definition(data_model)

    def test_add_data_different_types(self):
        data_model = {"type": mockClass, "value_id": {"k": int},
                      "fields": {"a": numpy.int64, "b": numpy.ndarray, "c": uuid.UUID}}
        raised = False
        try:
            storage = CQLIface()
            # Register data models
            storage.check_definition(data_model)
        except:
            raised = True
        self.assertFalse(raised, 'Exception raised')

    def test_add_data_model_new(self):
        data_model = {"type": mockClass, "value_id": {"k": int}, "fields": {"a": str}}
        storage = CQLIface()
        # Register data models
        id = storage.add_data_model(data_model)
        self.assertTrue(storage.data_models_cache[id])

    def test_add_data_model_StorageObj(self):
        data_model = {"type": StorageObj, "value_id": {"k": int}, "fields": {"a": str}}
        storage = CQLIface()
        # Register data models
        id = storage.add_data_model(data_model)
        self.assertTrue(storage.data_models_cache[id])

    def test_add_data_model_StorageDict(self):
        data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": str}}
        storage = CQLIface()
        # Register data models
        id = storage.add_data_model(data_model)
        self.assertTrue(storage.data_models_cache[id])

    def test_add_data_model_StorageNumpy(self):
        data_model = {"type": StorageNumpy, "value_id": {"k": int}, "fields": {"a": str}}
        storage = CQLIface()
        # Register data models
        id = storage.add_data_model(data_model)
        self.assertTrue(storage.data_models_cache[id])

    def test_add_data_model_existing_one(self):
        data_model1 = {"type": mockClass, "value_id": {"k": int}, "fields": {"a": str}}
        data_model2 = {"type": mockClass, "value_id": {"k": int}, "fields": {"a": str}}
        storage = CQLIface()
        # Register data models
        storage.add_data_model(data_model1)
        id2 = storage.add_data_model(data_model1)
        self.assertTrue(storage.data_models_cache[id2])

    def test_add_data_model_complex_types(self):
        data_model = {"type": mockClass, "value_id": {"k": int, "k1": int},
                      "fields": {"a": {"value_id": {"k": int}, "fields": {"f": str}}}}
        storage = CQLIface()
        # Register data models
        id = storage.add_data_model(data_model)
        self.assertTrue(storage.data_models_cache[id])

    def test_add_data_model_complex_structure(self):
        data_model = {"type": mockClass, "value_id": {"k": int, "k1": [int, int]}, "fields": {"a": {
            "value_id": {"k": (int, str)}, "fields": {"f": str, "f2": (float,)}}}}
        storage = CQLIface()
        # Register data models
        id = storage.add_data_model(data_model)
        self.assertTrue(storage.data_models_cache[id])


if __name__ == "__main__":
    unittest.main()
