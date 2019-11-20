from collections import namedtuple
import numpy as np

from storage.cql_iface.tests.mockIStorage import IStorage


class StorageNumpy(IStorage, np.ndarray):

    _build_args = None
    _class_name = None
    _hcache_params = None
    _hcache = None
    _ksp = ""
    _table = ""
    _block_id = None

    args_names = ["storage_id", "class_name", "name", "shape", "dtype", "block_id", "built_remotely"]
    args = namedtuple('StorageNumpyArgs', args_names)

    def __new__(cls, input_array=None, storage_id=None, name=None, built_remotely=False, **kwargs):
        if input_array is None and name and storage_id is not None:
            result = cls.load_array(storage_id)
            input_array = result[0]
            obj = np.asarray(input_array).view(cls)
            obj._name = name
            obj.storage_id = storage_id
            obj._is_persistent = True
        elif not name and storage_id is not None:
            raise RuntimeError("hnumpy received storage id but not a name")
        elif (input_array is not None and name and storage_id is not None) \
                or (storage_id is None and name):
            obj = np.asarray(input_array).view(cls)
            obj.storage_id = storage_id
            obj._is_persistent = False
            obj.make_persistent(name)
        else:
            obj = np.asarray(input_array).view(cls)
            obj.storage_id = storage_id
            obj._is_persistent = storage_id is not None
        # Finally, we must return the newly created object:
        obj._built_remotely = built_remotely
        obj._class_name = '%s.%s' % (cls.__module__, cls.__name__)
        return obj

    def __init__(self, *args, **kwargs):
        super(StorageNumpy, self).__init__()

    @staticmethod
    def load_array(storage_id):
        #storage.StorageAPI.get_records(storage_id)
        '''
        (ksp, table) = extract_ks_table(name)
        hcache_params = (ksp, table,
                         storage_id, [], ['storage_id', 'cluster_id', 'block_id'],
                         [{'name': "payload", 'type': 'numpy'}],
                         {'cache_size': config.max_cache_size,
                          'writer_par': config.write_callbacks_number,
                          'write_buffer': config.write_buffer_size,
                          'timestamped_writes': config.timestamped_writes})
        hcache = HNumpyStore(*hcache_params)
        result = hcache.get_numpy([storage_id])
        if len(result) == 1:
            return [result[0], hcache, hcache_params]
        else:
            raise KeyError
        '''
        return [np.zeros(())]
