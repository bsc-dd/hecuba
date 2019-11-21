import datetime
import decimal
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
    _hecuba2cassandra_typemap = {
        None: 'NULL',
        bool: 'boolean',
        int: 'int',
        float: 'float',
        str: 'text',
        bytearray: 'blob',
        bytes: 'blob',
        tuple: 'tuple',
        frozenset: 'set',
        decimal.Decimal: 'decimal',
        datetime.date: 'date',
        datetime.datetime: 'datetime',
        datetime.time: 'time',
        numpy.int8: 'tinyint',
        numpy.int16: 'smallint',
        numpy.int64: 'double',
        numpy.ndarray: 'hecuba.hnumpy.StorageNumpy',
        numpy.unicode: 'varchar',
        uuid.UUID: 'uuid'
    }

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
                CQLIface._hecuba2cassandra_typemap[definition]
            except KeyError:
                raise TypeError(f"The type {definition} is not supported")

    def add_data_model(self, definition: dict) -> int:
        if not isinstance(definition, dict):
            raise TypeError("Expected a dict type as a definition")
        if not all(name in definition for name in ["type", "value_id", "fields"]):
            raise KeyError("Expected keys 'type', 'value_id' and 'fields'")
        if not (isinstance(definition["value_id"], dict) and isinstance(definition["fields"], dict)):
            raise TypeError("Expected keys 'value_id' and 'fields' to be dict")
        if not issubclass(definition["type"], IStorage):
            raise TypeError("Class must inherit IStorage")
        dict_definition = {k: definition[k] for k in ('value_id', 'fields')}
        CQLIface.check_values_from_definition(dict_definition)
        dm = list(definition.items())
        dm.sort()
        datamodel_id = hash(str(dm))
        try:
            self.data_models_cache[datamodel_id]
        except KeyError:
            self.data_models_cache[datamodel_id] = definition
            CqlCOMM.register_data_model(datamodel_id, definition)
        return datamodel_id
