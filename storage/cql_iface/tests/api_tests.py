import decimal
import unittest
import uuid
from typing import Tuple, NamedTuple

import numpy

from storage.cql_iface.cql_comm import config
from storage.cql_iface.cql_iface import CQLIface
from hecuba.IStorage import IStorage
from storage.cql_iface.tests.mockStorageObj import StorageObj
from storage.cql_iface.tests.mockhdict import StorageDict
from storage.cql_iface.tests.mockhnumpy import StorageNumpy


class TestClass(IStorage):

    def __new__(cls, name='', *args, **kwargs):
        toret = super(TestClass, cls).__new__(cls, name)
        return toret

    def __init__(self, *args, **kwargs):
        super(TestClass, self).__init__()


class TestClass2(IStorage):

    def __new__(cls, *args, name='', **kwargs):
        toret = super(TestClass2, cls).__new__(cls, name)
        return toret

    def __init__(self, *args, **kwargs):
        super(TestClass2, self).__init__()


class mockClass(IStorage):
    pass


class mockClassNoInherit:
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
            storage.add_data_model(data_model)

    def test_add_data_model_except_invalid_class(self):
        with self.assertRaises(TypeError):
            data_model = {"type": "a", "value_id": {"k": str}, "fields": {"a": str}}
            storage = CQLIface()
            # Register data models
            storage.add_data_model(data_model)

    def test_add_data_model_except_invalid_format(self):
        with self.assertRaises(KeyError):
            data_model = {"type": mockClass, "": {"k": str}, "fields": {"a": str}}
            storage = CQLIface()
            # Register data models
            storage.add_data_model(data_model)

    def test_add_data_model_except_not_value_fields_dict_type(self):
        with self.assertRaises(TypeError):
            data_model = {"type": mockClass, "value_id": [str], "fields": {"a": str}}
            storage = CQLIface()
            # Register data models
            storage.add_data_model(data_model)

    def test_add_data_model_except_incorrect_type(self):
        with self.assertRaises(TypeError):
            data_model = {"type": mockClass, "value_id": {"k": dict}, "fields": {"a": str}}
            storage = CQLIface()
            # Register data models
            storage.add_data_model(data_model)

    def test_add_data_model_except_incorrect_value_id(self):
        with self.assertRaises(TypeError):
            data_model = {"type": StorageObj, "value_id": {"k": dict}, "fields": {"a": str}}
            storage = CQLIface()
            # Register data models
            storage.add_data_model(data_model)

    def test_add_data_model_except_incorrect_inheritance(self):
        with self.assertRaises(TypeError):
            data_model = {"type": mockClassNoInherit, "value_id": {"k": dict}, "fields": {"a": str}}
            storage = CQLIface()
            # Register data models
            storage.add_data_model(data_model)

    def test_add_data_different_types(self):
        data_model = {"type": mockClass, "value_id": {"k": int},
                      "fields": {"a": numpy.int64, "b": numpy.ndarray, "c": uuid.UUID}}
        raised = False
        try:
            storage = CQLIface()
            # Register data models
            storage.add_data_model(data_model)
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
        data_model = {"type": StorageObj, "value_id": {"k": uuid.UUID}, "fields": {"a": str}}
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
        self.assertTrue(id2 in storage.data_models_cache)

    def test_add_data_model_complex_types(self):
        data_model = {"type": mockClass, "value_id": {"k": decimal.Decimal, "k1": numpy.ndarray},
                      "fields": {"k": numpy.unicode, "f": numpy.int64}}
        storage = CQLIface()
        # Register data models
        id = storage.add_data_model(data_model)
        self.assertTrue(storage.data_models_cache[id])

    def test_add_data_model_complex_structure(self):
        data_model = {"type": mockClass,
                      "value_id": {"k": int, "k1": [int], "k2": (int, uuid.UUID), "k3": {"a": float, "b": (bool, int)}},
                      "fields": {"f": [str, (str, str)]}}
        storage = CQLIface()
        # Register data models
        id = storage.add_data_model(data_model)
        self.assertTrue(storage.data_models_cache[id])

    def test_register_persistent_except_data_model_id_none(self):
        with self.assertRaises(ValueError):
            # Setup object
            given_name = 'storage_test.custom_obj'
            obj = TestClass(name=given_name)
            data_model = {"type": mockClass, "value_id": {"k": str}, "fields": {"a": int}}

            # Setup persistent storage
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(None, obj)

    def test_register_persistent_except_data_model_obj_not_istorage(self):
        with self.assertRaises(RuntimeError):
            # Setup object
            obj = mockClassNoInherit()
            data_model = {"type": mockClass, "value_id": {"k": str}, "fields": {"a": int}}
            # Setup persistent storage
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(None, obj)

    def test_register_persistent_except_data_model_obj_not_persistent(self):
        with self.assertRaises(ValueError):
            # Setup object
            obj = mockClass()
            data_model = {"type": mockClass, "value_id": {"k": str}, "fields": {"a": int}}
            # Setup persistent storage
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(None, obj)

    def test_register_persistent_except_data_model_not_registered(self):
        with self.assertRaises(KeyError):
            # Setup object
            given_name = 'storage_test.custom_obj'
            obj = TestClass(name=given_name)
            data_model = {"type": mockClass, "value_id": {"k": str}, "fields": {"a": int}}

            # Setup persistent storage
            storage = CQLIface()
            storage.register_persistent_object(8, obj)

    def test_register_persistent_obj_dict_ints(self):
        given_name = 'storage_test.custom_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))
        obj = TestClass(name=given_name)
        myid = obj.getID()
        name = obj.get_name()
        data_model = {"type": TestClass, "value_id": {"k": str}, "fields": {"a": int}}

        # Setup persistent storage
        storage = CQLIface()

        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)
        res = config.session.execute("SELECT * FROM hecuba.istorage WHERE storage_id={}".format(myid)).one()
        self.assertEqual(res.table_name, name)
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

    def test_register_persistent_obj_storage_obj(self):
        given_name = 'storage_test.custom_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))
        obj = TestClass(name=given_name)
        myid = obj.getID()
        name = obj.get_name()
        data_model = {"type": StorageObj, "value_id": {"k": uuid.UUID}, "fields": {"a": str, "b": str}}

        # Setup persistent storage
        storage = CQLIface()

        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)
        res = config.session.execute("SELECT * FROM hecuba.istorage WHERE storage_id={}".format(myid)).one()
        self.assertEqual(res.table_name, name)
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

    def test_register_persistent_obj_storage_obj_tuple(self):
        given_name = 'storage_test.custom_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))
        obj = TestClass(name=given_name)
        myid = obj.getID()
        name = obj.get_name()
        data_model = {"type": StorageObj, "value_id": {"k": uuid.UUID}, "fields": {"a": Tuple[int], "b": str}}

        # Setup persistent storage
        storage = CQLIface()

        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)
        res = config.session.execute("SELECT * FROM hecuba.istorage WHERE storage_id={}".format(myid)).one()
        self.assertEqual(res.table_name, name)
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

    def test_register_persistent_obj_storage_dict(self):
        given_name = 'storage_test.custom_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))
        obj = TestClass(name=given_name)
        myid = obj.getID()
        name = obj.get_name()
        data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": str, "b": str}}

        # Setup persistent storage
        storage = CQLIface()

        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)
        res = config.session.execute("SELECT * FROM hecuba.istorage WHERE storage_id={}".format(myid)).one()
        self.assertEqual(res.table_name, name)
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

    def test_put_record_except_invalid_uuid(self):
        with self.assertRaises(ValueError):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            data_model = {"type": StorageObj, "value_id": {"k": uuid.UUID}, "fields": {"a": int, "b": str, "c": float}}
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)
            storage.put_record("", None, None)

    def test_put_record_except_hcache_not_registered(self):
        with self.assertRaises(ValueError):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            data_model = {"type": StorageObj, "value_id": {"k": uuid.UUID}, "fields": {"a": int, "b": str, "c": float}}
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.put_record("", None, None)

    def test_put_record_except_key_value_must_be_ordered_dict(self):
        with self.assertRaises(TypeError):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageObj, "value_id": {"k": uuid.UUID}, "fields": {"a": int, "b": str, "c": float}}
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)
            storage.put_record(myid, None, None)

    def test_put_record_except_key_not_exist(self):
        with self.assertRaises(KeyError):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageObj, "value_id": {"k": uuid.UUID}, "fields": {"a": int, "b": str, "c": float}}
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)

            keys = NamedTuple('keys', [('k', uuid.UUID)])
            keys = keys(myid)._asdict()
            fields = NamedTuple('fields', [('a', int), ('b', str), ('d', float)])
            fields = fields(4, 'ab', 3.4)._asdict()
            storage.put_record(myid, keys, fields)

    def test_put_record_except_values_not_same_class(self):
        with self.assertRaises(Exception):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageObj, "value_id": {"k": uuid.UUID}, "fields": {"a": int, "b": str, "c": float}}
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)

            keys = NamedTuple('keys', [('k', uuid.UUID)])
            keys = keys(myid)._asdict()
            fields = NamedTuple('fields', [('a', int), ('b', str), ('c', int)])
            fields = fields(4, 'ab', 3)._asdict()
            storage.put_record(myid, keys, fields)

    def test_put_record_StorageObj(self):
        given_name = 'storage_test.complex_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

        obj = TestClass2(name=given_name)
        myid = obj.getID()
        data_model = {"type": StorageObj, "value_id": {"k": uuid.UUID}, "fields": {"a": int, "b": str, "c": float}}
        given_name = 'storage_test.dict'
        storage = CQLIface()
        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)

        keys = NamedTuple('keys', [('k', uuid.UUID)])
        keys = keys(myid)._asdict()
        fields = NamedTuple('fields', [('a', int), ('b', str), ('c', float)])
        fields = fields(44, 'zasd', 8.0)._asdict()

        storage.put_record(myid, keys, fields)
        hcache = storage.hcache_by_id[myid]

        returned_values = hcache.get_row([myid])

        self.assertEqual(len([44, 'zasd', 8.0]), len(returned_values))

        for val, ret_val in zip([44, 'zasd', 8.0], returned_values):
            self.assertAlmostEqual(val, ret_val)

    def test_put_record_StorageObj_val_nones(self):
        given_name = 'storage_test.complex_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

        obj = TestClass2(name=given_name)
        myid = obj.getID()
        data_model = {"type": StorageObj, "value_id": {"k": uuid.UUID}, "fields": {"a": int, "b": str, "c": float}}
        given_name = 'storage_test.dict'
        storage = CQLIface()
        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)

        keys = NamedTuple('keys', [('k', uuid.UUID)])
        keys = keys(myid)._asdict()
        fields = NamedTuple('fields', [('a', int), ('b', str), ('c', float)])
        fields = fields(None, None, None)._asdict()
        storage.put_record(myid, keys, fields)

        returned_values = []
        hcache = storage.hcache_by_id[myid]
        for key in keys.values():
            returned_values.append(hcache.get_row([key]))

        self.assertEqual(len(fields.values()), len(returned_values[0]))

        for val, ret_val in zip(fields.values(), returned_values[0]):
            self.assertAlmostEqual(val, ret_val)

    def test_put_record_except_values_fields_not_same_size_as_data_model(self):
        with self.assertRaises(Exception):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": float}}
            given_name = 'storage_test.dict'
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)

            keys = NamedTuple('keys', [('k', int)])
            keys = keys(8)._asdict()
            fields = NamedTuple('fields', [('a', int), ('b', 'name'), ('c', float), ('d', str)])
            fields = fields(4, 'hola', 3.8, 'adeu')._asdict()

            storage.put_record(myid, keys, fields)

            returned_values = []
            hcache = storage.hcache_by_id[myid]
            for key in keys.values():
                returned_values.append(hcache.get_row([key]))

            self.assertEqual(len(fields.values()), len(returned_values[0]))

            for val, ret_val in zip(fields.values(), returned_values[0]):
                self.assertAlmostEqual(val, ret_val)

    def test_put_record_except_keys_not_mach_data_model_type(self):
        with self.assertRaises(Exception):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": float}}
            given_name = 'storage_test.dict'
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)

            keys = NamedTuple('keys', [('k', int)])
            keys = keys(8)._asdict()
            fields = NamedTuple('fields', [('z', int), ('b', 'name'), ('c', float)])
            fields = fields(4, 'hola', 3.8)._asdict()

            storage.put_record(myid, keys, fields)

            returned_values = []
            hcache = storage.hcache_by_id[myid]
            for key in keys.values():
                returned_values.append(hcache.get_row([key]))

            self.assertEqual(len(fields.values()), len(returned_values[0]))

            for val, ret_val in zip(fields.values(), returned_values[0]):
                self.assertAlmostEqual(val, ret_val)

    def test_put_record_except_values_not_mach_data_model_type(self):
        with self.assertRaises(Exception):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": float}}
            given_name = 'storage_test.dict'
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)

            keys = NamedTuple('keys', [('k', int)])
            keys = keys(8)._asdict()
            fields = NamedTuple('fields', [('a', int), ('b', 'name'), ('c', float)])
            fields = fields(4, 'hola', 'adeu')._asdict()

            storage.put_record(myid, keys, fields)

            returned_values = []
            hcache = storage.hcache_by_id[myid]
            for key in keys.values():
                returned_values.append(hcache.get_row([key]))

            self.assertEqual(len(fields.values()), len(returned_values[0]))

            for val, ret_val in zip(fields.values(), returned_values[0]):
                self.assertAlmostEqual(val, ret_val)

    def test_put_record_except_values_not_mach_data_model_type(self):
        with self.assertRaises(KeyError):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": float}}
            given_name = 'storage_test.dict'
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)

            keys = NamedTuple('keys', [('k', int)])
            keys = keys(8)._asdict()
            fields = NamedTuple('fields', [('a', int), ('b', 'name'), ('c', float)])
            fields = fields(4, 'hola', 'adeu')._asdict()

            storage.put_record(uuid.UUID('123e4567-0000-0000-0000-426655440000'), keys, fields)

    def test_put_record_StorageDict(self):
        given_name = 'storage_test.complex_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

        obj = TestClass(name=given_name)
        myid = obj.getID()
        data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": float}}
        given_name = 'storage_test.dict'
        storage = CQLIface()
        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)

        keys = NamedTuple('keys', [('k', int)])
        keys = keys(8)._asdict()
        fields = NamedTuple('fields', [('a', int), ('b', 'name'), ('c', float)])
        fields = fields(4, 'hola', 3.8)._asdict()

        storage.put_record(myid, keys, fields)

        returned_values = []
        hcache = storage.hcache_by_id[myid]
        for key in keys.values():
            returned_values.append(hcache.get_row([key]))

        self.assertEqual(len(fields.values()), len(returned_values[0]))

        for val, ret_val in zip(fields.values(), returned_values[0]):
            self.assertAlmostEqual(val, ret_val)

    def test_put_record_StorageDict_with_val_nones(self):
        given_name = 'storage_test.complex_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

        obj = TestClass(name=given_name)
        myid = obj.getID()
        data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": float}}
        given_name = 'storage_test.dict'
        storage = CQLIface()
        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)

        keys = NamedTuple('keys', [('k', int)])
        keys = keys(8)._asdict()
        fields = NamedTuple('fields', [('a', int), ('b', 'name'), ('c', float)])
        fields = fields(None, None, None)._asdict()

        storage.put_record(myid, keys, fields)

        returned_values = []
        hcache = storage.hcache_by_id[myid]
        for key in keys.values():
            returned_values.append(hcache.get_row([key]))

        self.assertEqual(len(fields.values()), len(returned_values[0]))

        for val, ret_val in zip(fields.values(), returned_values[0]):
            self.assertAlmostEqual(val, ret_val)

    def test_register_persistent_obj_storage_numpy(self):
        given_name = 'storage_test.custom_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))
        obj = TestClass(name=given_name)
        myid = obj.getID()
        name = obj.get_name()
        data_model = {"type": StorageNumpy, "value_id": {"k": int}, "fields": {"a": str, "b": str}}

        # Setup persistent storage
        storage = CQLIface()

        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)
        res = config.session.execute("SELECT * FROM hecuba.istorage WHERE storage_id={}".format(myid)).one()
        self.assertEqual(res.table_name, name)

    def test_get_record_except_invalid_uuid(self):
        with self.assertRaises(ValueError):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": float}}
            given_name = 'storage_test.dict'
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)

            keys = NamedTuple('keys', [('k', int)])
            keys = keys(8)._asdict()
            fields = NamedTuple('fields', [('a', int), ('b', 'name'), ('c', float)])
            fields = fields(4, 'hola', 3.8)._asdict()

            storage.put_record(myid, keys, fields)
            keys = NamedTuple('keys', [('k', str)])
            keys = keys('hola')._asdict()
            storage.get_record('invalid', keys)

    def test_get_record_except_hcache_not_registered(self):
        with self.assertRaises(KeyError):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))
            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": float}}
            storage = CQLIface()
            storage.hcache_by_id.pop(myid, None)
            keys = NamedTuple('keys', [('k', int)])
            keys = keys(8)._asdict()
            storage.get_record(myid, keys)

    def test_get_record_except_invalid_keys(self):
        with self.assertRaises(ValueError):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": float}}
            given_name = 'storage_test.dict'
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)

            keys = NamedTuple('keys', [('k', int)])
            keys = keys(8)._asdict()
            fields = NamedTuple('fields', [('a', int), ('b', 'name'), ('c', float)])
            fields = fields(4, 'hola', 3.8)._asdict()

            storage.put_record(myid, keys, fields)
            keys = NamedTuple('keys', [('k', str)])
            keys = keys('hola')._asdict()
            storage.get_record(myid, None)

    def test_get_record_except_invalid_keys_size(self):
        given_name = 'storage_test.complex_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

        obj = TestClass(name=given_name)
        myid = obj.getID()
        data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": float}}
        given_name = 'storage_test.dict'
        storage = CQLIface()
        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)

        keys = NamedTuple('keys', [('k', int)])
        keys = keys(8)._asdict()
        fields = NamedTuple('fields', [('a', int), ('b', 'name'), ('c', float)])
        fields = fields(4, 'hola', 3.8)._asdict()

        storage.put_record(myid, keys, fields)
        keys = NamedTuple('keys', [('k', str), ('a', str)])
        keys = keys('hola', 'abc')._asdict()
        result = storage.get_record(myid, keys)
        self.assertEqual(result, [])

    def test_get_record(self):
        given_name = 'storage_test.complex_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

        obj = TestClass(name=given_name)
        myid = obj.getID()
        data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": int}}
        given_name = 'storage_test.dict'
        storage = CQLIface()
        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)

        keys = NamedTuple('keys', [('k', int)])
        keys = keys(8)._asdict()
        fields = NamedTuple('fields', [('a', int), ('b', str), ('c', int)])
        fields = fields(4, 'hola', 4)._asdict()

        storage.put_record(myid, keys, fields)
        keys = NamedTuple('keys', [('k', int)])
        keys = keys(8)._asdict()
        result = storage.get_record(myid, keys)
        self.assertEqual(result, [4, 'hola', 4])

    def test_get_record_tuple(self):
        given_name = 'storage_test.complex_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

        obj = TestClass(name=given_name)
        myid = obj.getID()
        data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": Tuple[int, int], "c": int}}
        given_name = 'storage_test.dict'
        storage = CQLIface()
        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)

        keys = NamedTuple('keys', [('k', int)])
        keys = keys(8)._asdict()
        fields = NamedTuple('fields', [('a', int), ('b', Tuple[int, int]), ('c', int)])
        fields = fields(4, (6, 6), 4)._asdict()

        storage.put_record(myid, keys, fields)
        keys = NamedTuple('keys', [('k', int)])
        keys = keys(8)._asdict()
        result = storage.get_record(myid, keys)
        self.assertEqual(result, [4, (6, 6), 4])

    def test_get_record_except_key_not_uuid(self):
        with self.assertRaises(Exception):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageObj, "value_id": {"k": uuid.UUID}, "fields": {"a": int, "b": str, "c": int}}
            given_name = 'storage_test.dict'
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)

            keys = NamedTuple('keys', [('k', uuid.UUID)])
            keys = keys(myid)._asdict()
            fields = NamedTuple('fields', [('a', int), ('b', str), ('c', int)])
            fields = fields(4, 'abc', 4)._asdict()

            storage.put_record(myid, keys, fields)
            keys = NamedTuple('keys', [('k', uuid.UUID)])
            keys = keys('abc')._asdict()
            result = storage.get_record(myid, keys)
            self.assertEqual(result, [4, 'abc', 4])

    def test_get_record_so(self):
        given_name = 'storage_test.complex_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

        obj = TestClass(name=given_name)
        myid = obj.getID()
        data_model = {"type": StorageObj, "value_id": {"k": uuid.UUID}, "fields": {"a": int, "b": str, "c": int}}
        given_name = 'storage_test.dict'
        storage = CQLIface()
        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)

        keys = NamedTuple('keys', [('k', uuid.UUID)])
        keys = keys(myid)._asdict()
        fields = NamedTuple('fields', [('a', int), ('b', str), ('c', int)])
        fields = fields(4, 'abc', 4)._asdict()

        storage.put_record(myid, keys, fields)
        keys = NamedTuple('keys', [('k', uuid.UUID)])
        keys = keys(myid)._asdict()
        result = storage.get_record(myid, keys)
        self.assertEqual(result, [4, 'abc', 4])

    def test_get_record_tuple_so(self):
        given_name = 'storage_test.complex_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

        obj = TestClass(name=given_name)
        myid = obj.getID()
        data_model = {"type": StorageObj, "value_id": {"k": uuid.UUID},
                      "fields": {"a": int, "b": Tuple[int], "c": int}}
        given_name = 'storage_test.dict'
        storage = CQLIface()
        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)

        keys = NamedTuple('keys', [('k', uuid.UUID)])
        keys = keys(myid)._asdict()
        fields = NamedTuple('fields', [('a', int), ('b', Tuple[int]), ('c', int)])
        fields = fields(4, (6,), 4)._asdict()

        storage.put_record(myid, keys, fields)
        keys = NamedTuple('keys', [('k', uuid.UUID)])
        keys = keys(myid)._asdict()
        result = storage.get_record(myid, keys)
        self.assertEqual(result, [4, (6, 6), 4])

    def test_put_record_StorageDict_split_except_uuid_wrong_format(self):
        with self.assertRaises(ValueError):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageDict, "value_id": {"k1": int}, "fields": {"a": int, "b": str, "c": float}}
            given_name = 'storage_test.dict'
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)

            keys = NamedTuple('keys', [('k1', int)])
            keys = keys(8)._asdict()
            fields = NamedTuple('fields', [('a', int), ('b', 'name'), ('c', float)])
            fields = fields(None, None, None)._asdict()

            storage.put_record(myid, keys, fields)
            for partition in storage.split(4, 8):
                for val in partition.keys():
                    print(val)

    def test_put_record_StorageDict_split_except_subsets_wrong_type(self):
        with self.assertRaises(TypeError):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageDict, "value_id": {"k1": int}, "fields": {"a": int, "b": str, "c": float}}
            given_name = 'storage_test.dict'
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)

            keys = NamedTuple('keys', [('k1', int)])
            keys = keys(8)._asdict()
            fields = NamedTuple('fields', [('a', int), ('b', 'name'), ('c', float)])
            fields = fields(None, None, None)._asdict()

            storage.put_record(myid, keys, fields)
            for partition in storage.split(myid, 4.8):
                for val in partition.keys():
                    print(val)

    def test_put_record_StorageDict_split_and_get_data_locality_except_None(self):
        with self.assertRaises(ValueError):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": float}}
            given_name = 'storage_test.dict'
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)

            keys = NamedTuple('keys', [('k', int)])
            keys = keys(8)._asdict()
            fields = NamedTuple('fields', [('a', int), ('b', 'name'), ('c', float)])
            fields = fields(1, 'a', 3.0)._asdict()

            storage.put_record(myid, keys, fields)
            parts = []
            for partition in storage.split(myid, 9):
                parts.append(partition)
            self.assertTrue(storage.get_data_locality(None))

    def test_put_record_StorageDict_split_and_get_data_locality_except(self):
        with self.assertRaises(ValueError):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": float}}
            given_name = 'storage_test.dict'
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)

            keys = NamedTuple('keys', [('k', int)])
            keys = keys(8)._asdict()
            fields = NamedTuple('fields', [('a', int), ('b', 'name'), ('c', float)])
            fields = fields(1, 'a', 3.0)._asdict()

            storage.put_record(myid, keys, fields)
            parts = []
            for partition in storage.split(myid, 9):
                parts.append(partition)
            self.assertTrue(storage.get_data_locality(myid))

    def test_put_record_StorageDict_split_and_get_data_locality_except_wrong_UUID(self):
        with self.assertRaises(ValueError):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": float}}
            given_name = 'storage_test.dict'
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)

            keys = NamedTuple('keys', [('k', int)])
            keys = keys(8)._asdict()
            fields = NamedTuple('fields', [('a', int), ('b', 'name'), ('c', float)])
            fields = fields(1, 'a', 3.0)._asdict()

            storage.put_record(myid, keys, fields)
            parts = []
            for partition in storage.split(uuid.UUID('123e4567-e89b-12d3-a456-426655440000'), 9):
                parts.append(partition)
            self.assertTrue(storage.get_data_locality(myid))

    def test_put_record_StorageDict_split_and_get_data_locality_except_wrong_UUID_2(self):
        with self.assertRaises(TypeError):
            given_name = 'storage_test.complex_obj'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            obj = TestClass(name=given_name)
            myid = obj.getID()
            data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": float}}
            given_name = 'storage_test.dict'
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)
            storage.register_persistent_object(data_model_id, obj)

            keys = NamedTuple('keys', [('k', int)])
            keys = keys(8)._asdict()
            fields = NamedTuple('fields', [('a', int), ('b', 'name'), ('c', float)])
            fields = fields(1, 'a', 3.0)._asdict()

            storage.put_record(myid, {'k': '3'}, fields)

    def test_put_record_StorageDict_split_and_get_data_locality(self):
        given_name = 'storage_test.complex_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

        obj = TestClass(name=given_name)
        myid = obj.getID()
        data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": int, "b": str, "c": float}}
        given_name = 'storage_test.dict'
        storage = CQLIface()
        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)

        keys = NamedTuple('keys', [('k', int)])
        keys = keys(8)._asdict()
        fields = NamedTuple('fields', [('a', int), ('b', 'name'), ('c', float)])
        fields = fields(1, 'a', 3.0)._asdict()

        storage.put_record(myid, keys, fields)
        parts = []
        for partition in storage.split(myid, 9):
            parts.append(partition)
        self.assertTrue(storage.get_data_locality(parts[0]))

    def test_delete_persistent_except_object_id_not_uuid(self):
        with self.assertRaises(ValueError):
            given_name = 'storage_test.dict'
            config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

            # Setup object
            obj = TestClass(given_name)
            myid = obj.getID()
            name = obj.get_name()
            data_model = {"type": StorageNumpy, "value_id": {"k": int}, "fields": {"a": str, "b": str}}

            # Setup persistent storage
            storage = CQLIface()
            data_model_id = storage.add_data_model(data_model)

            storage.register_persistent_object(data_model_id, obj)

            storage.delete_persistent_object('exc')

    def test_delete_persistent_object_return_false(self):
        given_name = 'storage_test.dict'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

        # Setup object
        obj = TestClass(given_name)
        myid = obj.getID()
        name = obj.get_name()
        data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": str, "b": str}}

        # Setup persistent storage
        storage = CQLIface()
        data_model_id = storage.add_data_model(data_model)

        storage.register_persistent_object(data_model_id, obj)

        keys = NamedTuple('keys', [('k', int)])
        keys = keys(8)._asdict()
        fields = NamedTuple('fields', [('a', str), ('b', str)])
        fields = fields('a', 'a')._asdict()

        storage.put_record(myid, keys, fields)
        self.assertFalse(storage.delete_persistent_object(uuid.UUID('123e4567-0000-0000-0000-426655440000')))

    def test_delete_persistent_object_return_true(self):
        given_name = 'storage_test.dict'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

        # Setup object
        obj = TestClass(given_name)
        myid = obj.getID()
        name = obj.get_name()
        data_model = {"type": StorageDict, "value_id": {"k": int}, "fields": {"a": str, "b": str}}

        # Setup persistent storage
        storage = CQLIface()
        data_model_id = storage.add_data_model(data_model)

        storage.register_persistent_object(data_model_id, obj)

        keys = NamedTuple('keys', [('k', int)])
        keys = keys(8)._asdict()
        fields = NamedTuple('fields', [('a', str), ('b', str)])
        fields = fields('a', 'a')._asdict()

        storage.put_record(myid, keys, fields)
        self.assertTrue(storage.delete_persistent_object(myid))


if __name__ == "__main__":
    unittest.main()
