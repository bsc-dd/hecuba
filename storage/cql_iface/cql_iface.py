from ..storage_iface import StorageIface
from .cql_comm import CqlCOMM
import uuid
import numpy
from hecuba.IStorage import IStorage
from hecuba.storageiter import StorageIter

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

    def __init__(self):
        pass

    def add_data_model(self, definition):
        # datamodel_id
        dm = list(definition.items())
        dm.sort()
        data_model_id = hash(str(dm))
        self.data_models_cache[data_model_id] = definition
        CqlCOMM.register_data_model(data_model_id, definition)
        return data_model_id

    def register_persistent_object(self, datamodel_id, pyobject):
        if not isinstance(pyobject, IStorage):
            raise RuntimeError("Class does not inherit IStorage")

        object_id = pyobject.getID()
        data_model = self.data_models_cache[datamodel_id]
        # CQLIface.cache[object_id] = (datamodel_id, pyobject)
        object_name = pyobject.get_name()

        CqlCOMM.register_istorage(object_id, object_name, data_model)

        self.object_to_data_model[object_id] = datamodel_id

        if isinstance(pyobject, StorageIter):
            return self._replace_iterator(pyobject)

        CqlCOMM.create_table(object_id, object_name, data_model)

        # create hcache
        obj_class = pyobject.__class__.__name__
        if obj_class not in self.hcache_by_class:
            hc = CqlCOMM.create_hcache(object_id, object_name, data_model)

            self.hcache_by_class[obj_class] = hc
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
        results = []
        hcache = self.hcache_by_id[object_id]
        for key in key_list:
            try:
                results.append(hcache.get_row(key))
            except Exception:
                results.append([])

        return results

        # List < Value >

    def put_records(self, object_id, key_list, value_list):

        if not key_list:
            return

        for key, value in zip(key_list, value_list):
            self.hcache_by_id[object_id].put_row(key, value)

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

    def _replace_iterator(self, iter_obj):
        hc = self.hcache_by_name[iter_obj.get_name()]
        self.hcache_by_id[iter_obj.getID()] = hc

        class HcacheIterWrap:
            def __init__(self, myiter):
                self.myiter = myiter

            def __iter__(self):
                return self

            def __next__(self):
                return self.myiter.get_next()

        iter_obj.myiter = HcacheIterWrap(hc.iterkeys(100))
