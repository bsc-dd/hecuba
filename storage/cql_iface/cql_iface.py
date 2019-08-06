from ..storage_iface import StorageIface
from .cql_comm import CqlCOMM
import uuid
from hecuba.IStorage import IStorage

"""
Mockup on how the Cassandra implementation of the interface could work.
"""

class CQLIface(StorageIface):
    data_models_cache = {}
    object_to_data_model = {}
    hcache_by_name = {}
    hcache_by_class = {}
    hcache_by_id = {}

    def __init__(self):
        pass

    def add_data_model(self, definition):
        # datamodel_id
        data_model_id = hash(str(definition))
        self.data_models_cache[data_model_id] = definition
        CqlCOMM.register_data_model(data_model_id, definition)
        return data_model_id

    def register_persistent_object(self, datamodel_id, pyobject):
        if not isinstance(pyobject, IStorage):
            raise RuntimeError("Class does not inherit IStorage")

        object_id = pyobject.getID()

        #CQLIface.cache[object_id] = (datamodel_id, pyobject)

        CqlCOMM.register_istorage(object_id, pyobject.get_name(), self.data_models_cache[datamodel_id])
        dm = self.data_models_cache[datamodel_id]
        self.object_to_data_model[object_id] = datamodel_id


        if "primary_keys" in dm.keys():
            CqlCOMM.create_table(object_id, pyobject.get_name(), dm)

            hc = CqlCOMM.create_hcache(object_id, pyobject.get_name(), dm)

            self.hcache_by_name[pyobject.get_name()] = hc
            self.hcache_by_id[object_id] = hc
        elif "StorageIter" in dm.values():
            hc = self.hcache_by_name[pyobject.get_name()]
            self.hcache_by_id[object_id] = hc

            class HcacheIterWrap:
                def __init__(self, myiter):
                    self.myiter = myiter

                def __iter__(self):
                    return self

                def __next__(self):
                    return self.myiter.get_next()

            pyobject.myiter = HcacheIterWrap(hc.iterkeys(100))

        else:
            CqlCOMM.create_table(object_id, pyobject.get_name(), dm)

            # create hcache
            key = str(pyobject.__class__)
            if key not in self.hcache_by_class:
                dm = self.data_models_cache[datamodel_id]

                hc = CqlCOMM.create_hcache(object_id, pyobject.get_name(), dm)

                self.hcache_by_class[key] = hc
                self.hcache_by_name[pyobject.get_name()] = hc
                self.hcache_by_id[object_id] = hc

    def delete_persistent_object(self, object_id):
        try:
            CqlCOMM.delete_data(object_id)
        except KeyError:
            return False
        
        return True

    def add_index(self, datamodel_id):
        # IndexID
        raise NotImplemented("Add index not implemented yet")

    def get_records(self, object_id, key_list):
        # List < Value >
        results = []
        dm = self.data_models_cache[self.object_to_data_model[object_id]]
        persistent_res = self.hcache_by_id[object_id].get_row([object_id])

        for key in key_list:
            results.append(persistent_res[list(dm.keys()).index(key)])

        if not key_list:
            results = persistent_res

        return results

    def put_records(self, object_id, key_list, value_list):

        if not key_list:
            return

        # uint64[] ?
        toinsert = []
        dm = self.data_models_cache[self.object_to_data_model[object_id]]

        if "primary_keys" in dm.keys():
            for key, value in zip(key_list, value_list):
                self.hcache_by_id[object_id].put_row([key], value)

        else:
            for key in dm.keys():
                if key not in key_list:
                    toinsert.append(None)
                else:
                    toinsert.append(value_list[key_list.index(key)])


            self.hcache_by_id[object_id].put_row([object_id], toinsert)

        # obj_cache = CQLIface.records_cache.get(object_id, dict())
        # for key, value in zip(key_list, value_list):
        #     obj_cache[key] = value
        # CQLIface.records_cache[object_id] = obj_cache

    def split(self, object_id):
        # List < object_id >
        splits = []
        for i in range(0, 32):
            tmp_uid = uuid.uuid4()
            splits.append(tmp_uid)
            CQLIface.cache[tmp_uid] = (CQLIface.cache[object_id].__class__.datamodel_id, CQLIface.cache[object_id])
        return splits

    def get_data_locality(self, object_id):
        # List < Node >
        return ['127.0.0.1']
