from typing import Union, Tuple
from uuid import UUID

import numpy
import pickle
from hfetch import Hcache, HNumpyStore

from . import config
from .config import _hecuba2cassandra_typemap, log
from .queries import istorage_prepared_st, istorage_read_entry


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
    def register_istorage(obj_id: UUID, table_name: str, obj_name:str, data_model: dict) -> UUID:
        row = config.session.execute(istorage_read_entry, [obj_id])
        if not row:
            obj_info = [obj_id, table_name, obj_name, pickle.dumps(data_model)]
            config.execute(istorage_prepared_st, obj_info)
        return obj_id

    @staticmethod
    def register_data_model(data_model_id: int, definition: dict) -> None:
        # extract keys, values and so on
        pass

    @staticmethod
    def parse_definition_to_cass_format(fields_dict):
        all_values = ""
        for k, v in fields_dict.items():
            try:
                all_values = all_values + "%s %s," % (k, _hecuba2cassandra_typemap[v])
            except KeyError:
                if v.__origin__ in _hecuba2cassandra_typemap:
                    val = str(v)
                    all_values = all_values + str(k) + f" tuple<{val[val.find('[') + 1:val.rfind(']')]}>,"
        return all_values[:-1]

    @staticmethod
    def create_table(name: str, definition: dict) -> None:
        ksp, table = extract_ksp_table(name)
        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (ksp, config.replication)
        config.session.execute(query_keyspace)

        primary_keys = definition['value_id']
        columns = definition['fields']

        pks = str.join(',', primary_keys.keys())
        if definition["type"] is numpy.ndarray:
            pks = "(storage_id, cluster_id),block_id"

        all_keys = CqlCOMM.parse_definition_to_cass_format(primary_keys)
        if columns:
            all_cols = CqlCOMM.parse_definition_to_cass_format(columns)
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
            log.warn("Error creating the table: %s %s", query_table, ex)
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
        hcache_params = CqlCOMM.hcache_parameters_generator(ksp, table, object_id, list(definition["value_id"].keys()),
                                                            list(definition["fields"].keys()))
        if definition["type"] is numpy.ndarray:
            return HNumpyStore(*hcache_params)
        else:
            return Hcache(*hcache_params)
