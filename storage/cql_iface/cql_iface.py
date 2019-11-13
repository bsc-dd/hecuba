import uuid

import numpy
from storage.cql_iface.tests.mockIStorage import IStorage

from .cql_comm import CqlCOMM
from ..storage_iface import StorageIface

"""
Mockup on how the Cassandra implementation of the interface could work.
"""


class CQLIface(StorageIface):
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

    # User class to Cassandra data type
    _hecuba2cassandra_typemap = {str: 'text',
                                 bool: 'boolean',
                                 float: 'float',
                                 int: 'int',
                                 tuple: 'tuple',
                                 list: 'list',
                                 set: 'set',
                                 dict: 'map',
                                 bytearray: 'blob',
                                 bytes: 'blob',
                                 numpy.int64: 'double',
                                 numpy.ndarray: 'hecuba.hnumpy.StorageNumpy',
                                 uuid.UUID: 'uuid'}

    def __init__(self):
        pass

    @staticmethod
    def is_persistent(pyobject):
        return pyobject.getID() is not None and pyobject.get_name() is not None

    def check_definition(self, definition):
        if not isinstance(definition, dict):
            raise TypeError("Expected a dict type as a definition")
        elif not all(name in definition for name in ["type", "value_id", "fields"]):
            raise KeyError("Expected keys 'type', 'value_id' and 'fields'")
        elif not issubclass(definition["type"], IStorage):
            raise TypeError("Class must inherit IStorage")

    def add_data_model(self, definition):
        self.check_definition(definition)
        dm = list(definition.items())
        dm.sort()
        datamodel_id = hash(str(dm))
        try:
            self.data_models_cache[datamodel_id]
        except KeyError:
            self.data_models_cache[datamodel_id] = definition
            CqlCOMM.register_data_model(datamodel_id, definition)
        return datamodel_id
