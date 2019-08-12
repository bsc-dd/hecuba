import unittest

from storage.cql_iface.cql_iface import CQLIface
from storage.cql_iface.cql_comm import config

from hecuba import StorageDict
from hecuba.IStorage import IStorage
from hecuba.tools import storage_id_from_name


class ConcurrentDict(StorageDict):
    '''
    @TypeSpec dict <<key:int>, value0:int, value1:float>
    '''


class TestClass(IStorage):

    def __new__(cls, name='', *args, **kwargs):
        toret = super(TestClass, cls).__new__(cls)
        storage_id = kwargs.get('storage_id', None)
        if storage_id is None and name:
            storage_id = storage_id_from_name(name)

        if name or storage_id:
            toret.setID(storage_id)
            toret.set_name(name)
            toret._is_persistent = True
        return toret

    def __init__(self, *args, **kwargs):
        super(TestClass, self).__init__()


class HfetchTests(unittest.TestCase):
    def test_instantiate(self):
        result = CQLIface()
        self.assertIsNotNone(result)

    def test_register_dict_of_ints(self):
        given_name = 'storage_test.custom_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))
        # Setup object
        obj = TestClass(given_name)
        myid = obj.getID()
        name = obj.get_name()
        data_model = {"type": TestClass, "keys": {"k": str}, "cols": {"a": int}}

        # Setup persistent storage
        storage = CQLIface()
        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)

        res = config.session.execute("SELECT * FROM hecuba.istorage WHERE storage_id={}".format(myid)).one()
        self.assertEqual(res.name, name)
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

    def test_add_data_model(self):
        data_model_a = {"type": "typeA", "keys": {"k": str}, "cols": {"a": int}}
        data_model_copy = {"type": "typeA", "cols": {"a": int}, "keys": {"k": str}}
        data_model_b = {"type": "typeB", "keys": {"k": str}, "cols": {"a": int}}

        # Setup persistent storage
        storage = CQLIface()

        # Register data models
        id_a = storage.add_data_model(data_model_a)
        id_b = storage.add_data_model(data_model_b)
        id_a_cpy = storage.add_data_model(data_model_copy)

        # Compare the given ids
        self.assertEqual(data_model_a, data_model_copy)
        self.assertEqual(id_a, id_a_cpy)
        self.assertNotEqual(id_a, id_b)

    def test_del_persistent_object(self):
        given_name = 'storage_test.dict'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

        # Setup object
        obj = ConcurrentDict(given_name)
        myid = obj.getID()
        name = obj.get_name()
        data_model_id = obj._data_model_id

        self.assertEqual(given_name, name)

        # Setup persistent storage
        storage = CQLIface()

        storage.register_persistent_object(data_model_id, obj)

        res = config.session.execute("SELECT * FROM hecuba.istorage WHERE storage_id={}".format(myid)).one()
        self.assertEqual(res.name, name)

        for i in range(10):
            obj[i] = [i, float(i) / 10.0]

        del obj
        storage.delete_persistent_object(myid)

        res = config.session.execute("SELECT * FROM hecuba.istorage WHERE storage_id={}".format(myid)).one()
        self.assertIsNone(res)

        res = config.session.execute("SELECT count(*) FROM {}".format(name)).one()
        # TODO self.assertIsNone

        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

    def test_put_records(self):
        given_name = 'storage_test.complex_obj'
        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))

        obj = TestClass(given_name)
        myid = obj.getID()
        name = obj.get_name()
        data_model = {"type": "typeA", "keys": {"k": int}, "cols": {"a": int, "b": str, "c": float}}
        given_name = 'storage_test.dict'
        storage = CQLIface()
        data_model_id = storage.add_data_model(data_model)
        storage.register_persistent_object(data_model_id, obj)

        fields_ids = []
        values = []
        ninserts = 100
        for i in range(ninserts):
            fields_ids.append([i * 100])
            values.append([i, "someText{}".format(i), i / 10.0])

        storage.put_records(myid, fields_ids, values)

        returned_values = storage.get_records(myid, fields_ids)

        self.assertEqual(len(values), len(returned_values))

        for val, ret_val in zip(values, returned_values):
            self.assertEqual(val[0], ret_val[0])
            self.assertEqual(val[1], ret_val[1])
            self.assertAlmostEqual(val[2], ret_val[2], places=6)

        config.session.execute("DROP TABLE IF EXISTS {}".format(given_name))
