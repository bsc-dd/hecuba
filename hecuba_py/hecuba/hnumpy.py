from collections import namedtuple
import numpy as np
import uuid

from .IStorage import IStorage, AlreadyPersistentError

import storage


class StorageNumpy(IStorage, np.ndarray):
    class np_meta(object):
        def __init__(self, shape, dtype, block_id):
            self.dims = shape
            self.type = dtype
            self.block_id = block_id

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
        if storage_id:
            # already exists
            pass

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

    # used as copy constructor
    def __array_finalize__(self, obj):
        if obj is None:
            return
            # self.storage_id = getattr(obj, 'storage_id', None)
            # if self.storage_id and not hasattr(self, '_hcache'):
            #    self.make_persistent(obj.name)

    @staticmethod
    def load_array(storage_id):
        storage.StorageAPI.get_records(storage_id)
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

    def make_persistent(self, name):
        if self._is_persistent:
            raise AlreadyPersistentError("This StorageNumpy is already persistent [Before:{}][After:{}]",
                                         self._name, name)

        super().make_persistent(name)
        self._build_args = self.args(self.storage_id, self._class_name, self._name,
                                     self.shape, self.dtype.num, self._block_id, False)

        if self._data_model_id is None:
            data_model = {"type": type(self.view(np.ndarray)),
                          "keys": {"storage_id": uuid.UUID, "cluster_id": int, "block_id": int},
                          "cols": {"payload": bytearray}}
            self._data_model_id = storage.StorageAPI.add_data_model(data_model)

        storage.StorageAPI.register_persistent_object(datamodel_id=self._data_model_id, pyobject=self)
        # storage.StorageAPI.put_records(self.storage_id, [slice(None, None)], [self.view(np.ndarray)])
        storage.StorageAPI.put_records(self.storage_id, [[self.storage_id]], [[self.view(np.ndarray)]])

    def split(self):
        storage.StorageAPI.split(self.storage_id)

    def __iter__(self):
        return iter(self.view(np.ndarray))

    def __contains__(self, item):
        return item in self.view(np.ndarray)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = []
        for input_ in inputs:
            if isinstance(input_, StorageNumpy):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = kwargs.pop('out', None)
        if outputs:
            out_args = []
            for output in outputs:
                if isinstance(output, StorageNumpy):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super(StorageNumpy, self).__array_ufunc__(ufunc, method,
                                                            *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            return

        if self.storage_id and len(self.shape):
            self._hcache.save_numpy([self.storage_id], [self])

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((result
                         if output is None else output)
                        for result, output in zip(results, outputs))

        return results[0] if len(results) == 1 else results
