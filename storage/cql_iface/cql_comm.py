from typing import Union
from uuid import UUID

import numpy
from hfetch import Hcache, HNumpyStore

from . import config
from .config import _hecuba2cassandra_typemap, log
from .queries import istorage_prepared_st, istorage_read_entry
from .tests.mockStorageObj import StorageObj
from .tests.mockhdict import StorageDict


def extract_ksp_table(name):
    """
    Method used to obtain keyspace and table from a given name
    Args:
        name: a string containing keyspace name and table name, or only table name
    Returns:
        a tuple containing keyspace name and table name
    """

    try:
        ksp = name[:name.index('.')]
        table = name[len(ksp) + 1:]
    except ValueError:
        ksp = config.execution_name
        table = name
    return ksp.lower(), table.lower()


class CqlCOMM(object):

    @staticmethod
    def register_istorage(obj_id: UUID, obj_name: str, data_model: dict) -> UUID:
        row = config.session.execute(istorage_read_entry, [obj_id])
        if not row:
            obj_info = [obj_id, obj_name, str(data_model)]
            config.execute(istorage_prepared_st, obj_info)
        return obj_id

    @staticmethod
    def register_data_model(data_model_id: int, definition: dict) -> None:
        # extract keys, values and so on
        pass

    @staticmethod
    def create_table(name: str, definition: dict) -> None:
        # StorageObj for now
        ksp, table = extract_ksp_table(name)
        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (ksp, config.replication)
        config.session.execute(query_keyspace)

        primary_keys = definition['value_id']
        columns = definition['fields']

        pks = str.join(',', primary_keys.keys())
        if definition["type"] is numpy.ndarray:
            pks = "(storage_id, cluster_id),block_id"

        all_keys = ",".join("%s %s" % (k, _hecuba2cassandra_typemap[v]) for k, v in primary_keys.items())
        all_cols = ",".join("%s %s" % (k, _hecuba2cassandra_typemap[v]) for k, v in columns.items())
        if all_cols:
            total_cols = all_keys + ',' + all_cols
        else:
            total_cols = all_keys

        query_table = "CREATE TABLE IF NOT EXISTS %s.%s (%s, PRIMARY KEY (%s));" \
                      % (ksp,
                         table,
                         total_cols,
                         pks)
        try:
            log.debug('MAKE PERSISTENCE: %s', query_table)
            config.session.execute(query_table)
        except Exception as ex:
            log.warn("Error creating the StorageDict table: %s %s", query_table, ex)
            raise ex

    @staticmethod
    def hcache_parameters_generator(ksp: str, table: str, object_id: UUID, keys: list, columns: list) -> tuple:
        hcache_params = (ksp, table, object_id, [(-2 ** 63, 2 ** 63 - 1)], keys, columns,
                         {'cache_size': config.max_cache_size,
                          'writer_par': config.write_callbacks_number,
                          'writer_buffer': config.write_buffer_size,
                          'timestamped_writes': config.timestamped_writes})
        return hcache_params

    @staticmethod
    def create_hcache(object_id: UUID, name: str, definition: dict) -> Union[object, HNumpyStore, Hcache]:
        ksp, table = extract_ksp_table(name)
        if issubclass(definition.get("type", None), StorageObj):
            class HcacheWrapper(object):
                def __init__(self, definition, object_id, ksp, table):
                    self.internal_caches = {}
                    self.object_id = object_id
                    for col in definition["fields"].keys():
                        hc = Hcache(ksp, table, object_id, [(-2 ** 63, 2 ** 63 - 1)], list(definition["value_id"].keys()), [col],
                                    {'cache_size': config.max_cache_size,
                                     'writer_par': config.write_callbacks_number,
                                     'writer_buffer': config.write_buffer_size,
                                     'timestamped_writes': config.timestamped_writes})

                        self.internal_caches[col] = hc

                def get_row(self, attr):
                    return self.internal_caches[attr].get_row([self.object_id])[0]

                def put_row(self, attr, val):
                    self.internal_caches[attr].put_row([self.object_id], [val])

            return HcacheWrapper(definition, object_id, ksp, table)

        else:
            keys = [k for k in definition["value_id"].keys()]
            columns = [k for k in definition["fields"].keys()]
            hcache_params = CqlCOMM.hcache_parameters_generator(ksp, table, object_id, keys, columns)
            if definition["type"] is numpy.ndarray:
                return HNumpyStore(*hcache_params)
            elif issubclass(definition.get("type", None), StorageDict):
                return Hcache(*hcache_params)
