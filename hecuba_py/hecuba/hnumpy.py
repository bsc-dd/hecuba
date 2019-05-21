import uuid
from collections import namedtuple

import numpy as np
from hecuba import config, log
# from hfetch import Hcache
from hfetch import HNumpyStore

from IStorage import IStorage, AlreadyPersistentError, _extract_ks_tab


class StorageNumpy(np.ndarray, IStorage):
    class np_meta(object):
        def __init__(self, shape, dtype, block_id):
            self.dims = shape
            self.type = dtype
            self.block_id = block_id

    _storage_id = None
    _build_args = None
    _class_name = None
    _hcache_params = None
    _hcache = None
    _is_persistent = False
    _ksp = ""
    _table = ""
    _block_id = None
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba.istorage'
                                                  '(storage_id, class_name, name, numpy_meta)'
                                                  'VALUES (?,?,?,?)')

    args_names = ["storage_id", "class_name", "name", "shape", "dtype", "block_id", "built_remotely"]
    args = namedtuple('StorageNumpyArgs', args_names)

    def __new__(cls, input_array=None, storage_id=None, name=None, built_remotely=False, **kwargs):

        if input_array is None and name and storage_id is not None:
            result = cls.load_array(storage_id, name)
            input_array = result[0]
            obj = np.asarray(input_array).view(cls)
            obj._is_persistent = True
            (obj._ksp, obj._table) = _extract_ks_tab(name)
            obj._hcache = result[1]
            obj._hcache_params = result[2]
            obj._storage_id = storage_id
        elif not name and storage_id is not None:
            raise RuntimeError("hnumpy received storage id but not a name")
        elif (input_array is not None and name and storage_id is not None) \
                or (storage_id is None and name):
            obj = np.asarray(input_array).view(cls)
            obj._storage_id = storage_id
            obj._built_remotely = built_remotely
            obj.make_persistent(name)
        else:
            obj = np.asarray(input_array).view(cls)
            obj._storage_id = storage_id
        # Finally, we must return the newly created object:
        obj._built_remotely = built_remotely
        obj._class_name = '%s.%s' % (cls.__module__, cls.__name__)
        return obj

    # used as copy constructor
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._storage_id = getattr(obj, '_storage_id', None)

    @staticmethod
    def build_remotely(new_args):
        """
            Launches the StorageNumpy.__init__ from the uuid api.getByID
            Args:
                new_args: a list of all information needed to create again the StorageNumpy
            Returns:
                so: the created StorageNumpy
        """
        log.debug("Building StorageNumpy object with %s", new_args)
        return StorageNumpy(name=new_args.name, storage_id=new_args.storage_id)

    @staticmethod
    def _store_meta(storage_args):
        """
            Saves the information of the object in the istorage table.
            Args:.
                storage_args (object): contains all data needed to restore the object from the workers
        """
        log.debug("StorageObj: storing media %s", storage_args)
        try:
            config.session.execute(StorageNumpy._prepared_store_meta,
                                   [storage_args.storage_id, storage_args.class_name,
                                    storage_args.name, StorageNumpy.np_meta(storage_args.shape, storage_args.dtype,
                                                                            storage_args.block_id)])

        except Exception as ex:
            log.warn("Error creating the StorageNumpy metadata with args: %s" % str(storage_args))
            raise ex

    @staticmethod
    def load_array(storage_id, name):
        (ksp, table) = _extract_ks_tab(name)
        hcache_params = (ksp, table + '_numpies',
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

    def make_persistent(self, name):
        if self._is_persistent:
            raise AlreadyPersistentError("This StorageNumpy is already persistent [Before:{}.{}][After:{}]",
                                         self._ksp, self._table, name)
        self._is_persistent = True

        (self._ksp, self._table) = _extract_ks_tab(name)
        if self._storage_id is None:
            self._storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, self._ksp + '.' + self._table + '_numpies')

        self._build_args = self.args(self._storage_id, self._class_name, self._ksp + '.' + self._table,
                                     self.shape, self.dtype.num, self._block_id, self._built_remotely)

        if not self._built_remotely:
            log.info("PERSISTING DATA INTO %s %s", self._ksp, self._table)

            query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (self._ksp, config.replication)
            config.session.execute(query_keyspace)

            config.session.execute('CREATE TABLE IF NOT EXISTS ' + self._ksp + '.' + self._table + '_numpies'
                                                                                                   '(storage_id uuid , '
                                                                                                   'cluster_id int, '
                                                                                                   'block_id int, '
                                                                                                   'payload blob, '
                                                                                                   'PRIMARY KEY((storage_id,cluster_id),block_id))')

        self._hcache_params = (self._ksp, self._table + '_numpies',
                               self._storage_id, [], ['storage_id', 'cluster_id', 'block_id'],
                               [{'name': "payload", 'type': 'numpy'}],
                               {'cache_size': config.max_cache_size,
                                'writer_par': config.write_callbacks_number,
                                'write_buffer': config.write_buffer_size,
                                'timestamped_writes': config.timestamped_writes})

        self._hcache = HNumpyStore(*self._hcache_params)
        if len(self.shape) != 0:
            self._hcache.save_numpy([self._storage_id], [self])
        self._store_meta(self._build_args)

    def delete_persistent(self):
        """
            Deletes the Cassandra table where the persistent StorageObj stores data
        """
        query = "DELETE FROM %s.%s WHERE storage_id = %s;" % (self._ksp, self._table + '_numpies', self._storage_id)
        query2 = "DELETE FROM hecuba.istorage WHERE storage_id = %s;" % self._storage_id
        log.debug("DELETE PERSISTENT: %s", query)
        config.session.execute(query)
        config.session.execute(query2)
        self._is_persistent = False

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

        if self._is_persistent and len(self.shape):
            self._hcache.save_numpy([self._storage_id], [self])

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((result
                         if output is None else output)
                        for result, output in zip(results, outputs))

        return results[0] if len(results) == 1 else results
