from collections import namedtuple

import numpy as np
from . import config, log
from hfetch import HNumpyStore, HArrayMetadata

from .IStorage import IStorage
from .tools import extract_ks_tab, get_istorage_attrs


class StorageNumpy(IStorage, np.ndarray):
    _build_args = None
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba.istorage'
                                                  '(storage_id, class_name, name, numpy_meta)'
                                                  'VALUES (?,?,?,?)')

    args_names = ["storage_id", "class_name", "name", "metas"]
    args = namedtuple('StorageNumpyArgs', args_names)

    def __new__(cls, input_array=None, storage_id=None, name=None, **kwargs):
        if input_array is None and storage_id is not None:
            # Load metadata
            istorage_metas = get_istorage_attrs(storage_id)
            name = name or istorage_metas[0].name
            numpy_metadata = istorage_metas[0].numpy_meta

            # Load array
            hcache = StorageNumpy._create_hcache(name)
            input_array = hcache.get_numpy(storage_id, numpy_metadata)
            obj = np.asarray(input_array).view(cls)
            (obj._ksp, obj._table) = extract_ks_tab(name)
            obj._hcache = hcache
        else:
            obj = np.asarray(input_array).view(cls)

        # Finally, we must return the newly created object:
        obj._class_name = '%s.%s' % (cls.__module__, cls.__name__)
        return obj

    def __init__(self, input_array=None, storage_id=None, name=None, **kwargs):
        IStorage.__init__(self, storage_id=storage_id, name=name, **kwargs)
        if input_array is not None and (name or storage_id):
            self.make_persistent(name)

        metas = HArrayMetadata(list(self.shape), list(self.strides), self.dtype.kind, self.dtype.byteorder,
                               self.itemsize, self.flags.num, 0)

        self._build_args = self.args(self.storage_id, self._class_name, self._get_name(), metas)

    # used as copy constructor
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.storage_id = getattr(obj, "storage_id", None)
        self._hcache = getattr(obj, "_hcache", None)
        try:
            self._build_args = obj._build_args
        except AttributeError:
            self._build_args = HArrayMetadata(list(self.shape), list(self.strides), self.dtype.kind,
                                              self.dtype.byteorder, self.itemsize, self.flags.num, 0)
        try:
            name = obj._get_name()
            self._set_name(name)
        except AttributeError:
            pass

    @staticmethod
    def _create_tables(name):
        (ksp, table) = extract_ks_tab(name)
        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (ksp, config.replication)
        config.session.execute(query_keyspace)

        config.session.execute(
            'CREATE TABLE IF NOT EXISTS ' + ksp + '.' + table + '(storage_id uuid , '
                                                                'cluster_id int, '
                                                                'block_id int, '
                                                                'payload blob, '
                                                                'PRIMARY KEY((storage_id,cluster_id),block_id))')

    @staticmethod
    def _create_hcache(name):
        (ksp, table) = extract_ks_tab(name)
        hcache_params = (ksp, table,
                         {'cache_size': config.max_cache_size,
                          'writer_par': config.write_callbacks_number,
                          'write_buffer': config.write_buffer_size,
                          'timestamped_writes': config.timestamped_writes})

        return HNumpyStore(*hcache_params)

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
                                    storage_args.name, storage_args.metas])

        except Exception as ex:
            log.warn("Error creating the StorageNumpy metadata with args: %s" % str(storage_args))
            raise ex

    def make_persistent(self, name):
        super().make_persistent(name)

        if not self._built_remotely:
            self._create_tables(name)

        if not getattr(self, '_hcache', None):
            self._hcache = self._create_hcache(name)

        if None in self or not self.ndim:
            raise NotImplemented("Empty array persistance")

        hfetch_metas = HArrayMetadata(list(self.shape), list(self.strides), self.dtype.kind, self.dtype.byteorder,
                                      self.itemsize, self.flags.num, 0)

        self._build_args = self.args(self.storage_id, self._class_name, self._get_name(), hfetch_metas)
        if len(self.shape) != 0:
            self._hcache.save_numpy(self.storage_id, self, hfetch_metas)
        StorageNumpy._store_meta(self._build_args)

    def stop_persistent(self):
        super().stop_persistent()

        self.storage_id = None

    def delete_persistent(self):
        """
            Deletes the Cassandra table where the persistent StorageObj stores data
        """
        super().delete_persistent()
        query = "DELETE FROM %s.%s WHERE storage_id = %s;" % (self._ksp, self._table, self.storage_id)
        query2 = "DELETE FROM hecuba.istorage WHERE storage_id = %s;" % self.storage_id
        log.debug("DELETE PERSISTENT: %s", query)
        config.session.execute(query)
        config.session.execute(query2)
        self.storage_id = None

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
            self._hcache.save_numpy(self.storage_id, self, self._build_args.metas)

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((result
                         if output is None else output)
                        for result, output in zip(results, outputs))

        return results[0] if len(results) == 1 else results
