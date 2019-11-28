import itertools
from typing import List
from uuid import UUID

from storage.cql_iface.tests.mockhnumpy import StorageNumpy

from storage.cql_iface.tests.mockhdict import StorageDict

from storage.cql_iface.tests.mockIStorage import IStorage
from .config import _hecuba2cassandra_typemap
from .cql_comm import CqlCOMM
from ..storage_iface import StorageIface
from .tests.mockStorageObj import StorageObj

"""
Mockup on how the Cassandra implementation of the interface could work.
"""


class CQLIface(StorageIface):
    data_model_hcache = []
    # DataModelID - DataModelDef
    data_models_cache = {}
    # StorageID - DataModelID
    object_to_data_model = {}
    # Object Name - Cache
    hcache_by_name = {}
    # Object's class - Cache
    hcache_by_class = {}
    # StorageID - Cache
    hcache_by_id = {}

    def __init__(self):
        pass

    @staticmethod
    def check_values_from_definition(definition):
        if isinstance(definition, dict):
            for v in definition.values():
                CQLIface.check_values_from_definition(v)
        elif isinstance(definition, (list, set, tuple)):
            for v in definition:
                CQLIface.check_values_from_definition(v)
        else:
            try:
                _hecuba2cassandra_typemap[definition]
            except KeyError:
                raise TypeError(f"The type {definition} is not supported")

    def add_data_model(self, definition: dict) -> int:
        if not isinstance(definition, dict):
            raise TypeError("Expected a dict type as a definition")
        if not all(name in definition for name in ["type", "value_id", "fields"]):
            raise KeyError("Expected keys 'type', 'value_id' and 'fields'")
        if not (isinstance(definition["value_id"], dict) and isinstance(definition["fields"], dict)):
            raise TypeError("Expected keys 'value_id' and 'fields' to be dict")
        if definition["type"] is StorageObj and not all([definition["value_id"][k] is UUID for k in definition["value_id"].keys()]):
            raise TypeError("If the type is StorageObj the value_id values must be of type uuid")
        if not issubclass(definition["type"], IStorage):
            raise TypeError("Class must inherit IStorage")
        dm = sorted(definition.items())
        datamodel_id = hash(str(dm))
        try:
            self.data_models_cache[datamodel_id]
        except KeyError:
            dict_definition = {k: definition[k] for k in ('value_id', 'fields')}
            CQLIface.check_values_from_definition(dict_definition)
            self.data_models_cache[datamodel_id] = definition
            CqlCOMM.register_data_model(datamodel_id, definition)
        return datamodel_id

    def register_persistent_object(self, datamodel_id: int, pyobject: IStorage) -> UUID:
        if not isinstance(pyobject, IStorage):
            raise RuntimeError("Class does not inherit IStorage")
        elif not pyobject.is_persistent():
            raise ValueError("Class needs to be a persistent object, it needs id and name")
        elif datamodel_id is None:
            raise ValueError("datamodel_id cannot be None")
        try:
            data_model = self.data_models_cache[datamodel_id]
        except KeyError:
            raise KeyError("Before making a pyobject persistent, the data model needs to be registered")
        object_id = pyobject.getID()
         #TODO an object to data model can have more than 1 data model id, because the class and name can be the same one for different datamodels, for now we replace AND we change the name of the class in order to create hcache (another problem)
        self.object_to_data_model[object_id] = datamodel_id
        object_name = pyobject.get_name()
        CqlCOMM.register_istorage(object_id, object_name, data_model)
        CqlCOMM.create_table(object_name, data_model)
        obj_class = pyobject.__class__.__name__
        if datamodel_id not in self.data_model_hcache or obj_class not in self.hcache_by_class or object_name \
                not in self.hcache_by_name or not object_id in self.hcache_by_id:
            hc = CqlCOMM.create_hcache(object_id, object_name, data_model)
            self.hcache_by_class[obj_class] = hc
            self.hcache_by_name[object_name] = hc
            self.hcache_by_id[object_id] = hc
            self.data_model_hcache.append(datamodel_id)
        return object_id

    def put_record(self, object_id: UUID, key_list: List[object], value_list: List[object]) -> None:
        try:
            UUID(str(object_id))
        except ValueError:
            raise ValueError("The object_id is not an UUID")
        try:
            self.hcache_by_id[object_id]
        except KeyError:
            raise KeyError("hcache must be registered before in the function register_persistent_object")
        if not isinstance(key_list, list) and not isinstance(value_list, list):
            raise TypeError("key_list and value_list must be lists")
        if not (key_list and value_list):
            raise ValueError("key_list and value_list cannot be None")
        data_model = self.data_models_cache[self.object_to_data_model[object_id]]

        if issubclass(data_model["type"], StorageObj):
            if not len(key_list) == len(value_list):
                raise ValueError("key_list and value_list must have the same length")
            if not (all(x in data_model["fields"].keys() for x in key_list)):
                raise KeyError("value_list must have the keys that exist in the data model")
            dict_of_kv = dict(zip(key_list, value_list))
            if not all([isinstance(dict_of_kv[k], data_model["fields"][k]) for k in
                        key_list]):
                raise TypeError("the value types must be of the same class types as the data model")
            for key, value in zip(key_list, value_list):
                self.hcache_by_id[object_id].put_row(key, value)
        elif issubclass(data_model["type"], StorageDict):
            if len(key_list) != len(data_model["value_id"].keys()) or len(value_list) != len(data_model["fields"].keys()):
                raise ValueError("The length of the keys and values should be the same one as the data model definition")
            if not all([isinstance(ke, kc) for ke, kc in zip(key_list, data_model["value_id"].values())]):
                raise TypeError("The key types don't match the data model specification")
            if not all([isinstance(ke, kc) for ke, kc in zip(value_list, data_model["fields"].values())]):
                raise TypeError("The value types don't match the data model specification")
            self.hcache_by_id[object_id].put_row(key_list, value_list)

        elif issubclass(data_model["type"], StorageNumpy):
            raise NotImplemented("The class type is not supported")
        else:
            raise NotImplemented("The class type is not supported")