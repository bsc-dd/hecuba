import itertools as it
import uuid
from collections import namedtuple
from typing import Tuple

import numpy as np
from hfetch import HNumpyStore, HArrayMetadata

from . import config, log
from .IStorage import IStorage
from .tools import extract_ks_tab, get_istorage_attrs, storage_id_from_name


class StorageNumpy(IStorage, np.ndarray):
    _build_args = None
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba.istorage'
                                                  '(storage_id, table_name, obj_name, numpy_meta, block_id, base_numpy)'
                                                  'VALUES (?,?,?,?,?,?)')

    args_names = ["storage_id", "table_name", "obj_name", "numpy_meta", "block_id", "base_numpy"]
    args = namedtuple('StorageNumpyArgs', args_names)

    def np_split(self, block_size: Tuple[int, int]):
        # For now, only split in two dimensions is supported
        bn, bm = block_size
        for block_id, i in enumerate(range(0, self.shape[0], bn)):
            block = [self[i: i + bn, j:j + bm] for j in range(0, self.shape[1], bm)]
            obj = StorageNumpy(input_array=block, name=self.name, storage_id=uuid.uuid4(), block_id=block_id)
            yield obj

    def __new__(cls, input_array=None, name=None, storage_id=None, block_id=None, **kwargs):
        if input_array is None and (name is not None or storage_id is not None):
            if not storage_id:
                (ksp, table) = extract_ks_tab(name)
                name = ksp + "." + table
                storage_id = storage_id_from_name(name)

            # Load metadata
            istorage_metas = get_istorage_attrs(storage_id)
            name = name or istorage_metas[0].name
            numpy_metadata = istorage_metas[0].numpy_meta
            base_numpy = istorage_metas[0].base_numpy
            if base_numpy is not None:
                storage_id = base_numpy

            base_metas = get_istorage_attrs(storage_id)[0].numpy_meta

            # Load array
            result = cls.reserve_numpy_array(storage_id, name, base_metas)
            input_array = result[0]
            obj = np.asarray(input_array).view(cls)
            (obj._ksp, obj._table) = extract_ks_tab(name)
            obj._hcache = result[1]
            obj.storage_id = storage_id
            if base_numpy is not None:
                obj._partition_dims = numpy_metadata.dims
        else:
            obj = np.asarray(input_array).view(cls)

        obj._numpy_full_loaded = False
        obj._loaded_coordinates = []
        obj.name = name
        if getattr(obj, "_block_id", None) is None:
            obj._block_id = block_id
        # Finally, we must return the newly created object:
        obj._class_name = '%s.%s' % (cls.__module__, cls.__name__)
        return obj

    def __init__(self, input_array=None, name=None, storage_id=None, **kwargs):
        super(StorageNumpy, self).__init__()
        metas = HArrayMetadata(list(self.shape), list(self.strides), self.dtype.kind, self.dtype.byteorder,
                               self.itemsize, self.flags.num, 0)
        self._build_args = self.args(self.storage_id, self._class_name, self.get_name(), metas, self._block_id, None)

        if self.get_name() or self.storage_id:
            if input_array is not None:
                self.make_persistent(self.get_name())
            self._row_elem = self._hcache.get_elements_per_row(self.storage_id, metas)[0]
            self._is_persistent = True

    # used as copy constructor
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.storage_id = getattr(obj, 'storage_id', None)
        self._name = getattr(obj, '_name', None)
        self._hcache = getattr(obj, '_hcache', None)
        self._row_elem = getattr(obj, '_row_elem', None)
        self._loaded_coordinates = getattr(obj, '_loaded_coordinates', None)
        self._numpy_full_loaded = getattr(obj, '_numpy_full_loaded', None)
        self._is_persistent = getattr(obj, '_is_persistent', None)
        self._block_id = getattr(obj, '_block_id', None)
        try:
            self._build_args = obj._build_args
        except AttributeError:
            self._build_args = HArrayMetadata(list(self.shape), list(self.strides), self.dtype.kind,
                                              self.dtype.byteorder, self.itemsize, self.flags.num, 0)

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
                          'timestamped_writes': False})

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
                                    storage_args.name, storage_args.metas, storage_args.block_id,
                                    storage_args.base_numpy])

        except Exception as ex:
            log.warn("Error creating the StorageNumpy metadata with args: %s" % str(storage_args))
            raise ex

    @staticmethod
    def reserve_numpy_array(storage_id, name, metas):
        '''Provides a numpy array with the number of elements obtained through storage_id'''

        hcache = StorageNumpy._create_hcache(name)
        result = hcache.allocate_numpy(storage_id, metas)
        if len(result) == 1:
            return [result[0], hcache]
        else:
            raise KeyError

    def __getitem__(self, sliced_coord):
        log.info("RETRIEVING NUMPY")
        if self._is_persistent and not self._numpy_full_loaded:
            if sliced_coord == slice(None, None, None):
                new_coords = []
            else:
                try:
                    all_coords = np.argwhere(self.view(np.ndarray) == self.view(np.ndarray)).reshape(
                        *self.view(np.ndarray).shape, self.view(np.ndarray).ndim) // self._row_elem
                    new_coords = all_coords[sliced_coord].reshape(-1, self.view(np.ndarray).ndim)
                except IndexError:
                    return super(StorageNumpy, self).__getitem__(sliced_coord)
                new_coords = [tuple(coord) for coord in new_coords]
                new_coords = list(dict.fromkeys(new_coords))
            # coordinates is the union between the loaded coordiantes and the new ones
            coordinates = list(set(it.chain.from_iterable((self._loaded_coordinates, new_coords))))

            # checks if we already loaded the coordinates

            if ((len(coordinates) != len(self._loaded_coordinates)) and not self._numpy_full_loaded) or (
                    not self._numpy_full_loaded and not coordinates):
                if not coordinates:
                    self._numpy_full_loaded = True
                    new_coords = None

                self._hcache.load_numpy_slices([self.storage_id], self._build_args.metas, [self.base.view(np.ndarray)],
                                               new_coords)
                self._loaded_coordinates = coordinates
        return super(StorageNumpy, self).__getitem__(sliced_coord)

    def __setitem__(self, sliced_coord, values):
        log.info("WRITING NUMPY")
        if self._is_persistent:
            if sliced_coord == slice(None, None, None):
                new_coords = []
            else:
                try:
                    all_coords = np.argwhere(self.view(np.ndarray) == self.view(np.ndarray)).reshape(
                        *self.view(np.ndarray).shape, self.view(np.ndarray).ndim) // self._row_elem
                    new_coords = all_coords[sliced_coord].reshape(-1, self.view(np.ndarray).ndim)
                except IndexError:
                    return super(StorageNumpy, self).__getitem__(sliced_coord)
                new_coords = [tuple(coord) for coord in new_coords]
                new_coords = list(dict.fromkeys(new_coords))
            # coordinates is the union between the loaded coordiantes and the new ones
            coordinates = list(set(it.chain.from_iterable((self._loaded_coordinates, new_coords))))
            self._hcache.store_numpy_slices([self.storage_id], self._build_args.metas, [self.base.view(np.ndarray)],
                                            coordinates)
        return super(StorageNumpy, self).__setitem__(sliced_coord, values)

    def make_persistent(self, name):
        super().make_persistent(name)
        #
        # if not self._built_remotely:
        #     self._create_tables(name)
        #
        if not getattr(self, '_hcache', None):
             self._hcache = self._create_hcache(name)

        if None in self or not self.ndim:
             raise NotImplemented("Empty array persistance")

        hfetch_metas = HArrayMetadata(list(self.shape), list(self.strides), self.dtype.kind, self.dtype.byteorder,
                                      self.itemsize, self.flags.num, 0)
        self._build_args = self.args(self.storage_id, self._class_name, self.get_name(), hfetch_metas, self._block_id,
                                     None)

        if len(self.shape) != 0:
            self._hcache.store_numpy_slices([self.storage_id], self._build_args.metas, [self.base.view(np.ndarray)],
                                            None)
        StorageNumpy._store_meta(self._build_args)

    def stop_persistent(self):
        super().stop_persistent()

        self.storage_id = None

    def delete_persistent(self):
        """
            Deletes the Cassandra table where the persistent StorageObj stores data
        """
        super().delete_persistent()

        clusters_query = "SELECT cluster_id FROM %s WHERE storage_id = %s ALLOW FILTERING;" % (
            self._name, self.storage_id)
        clusters = config.session.execute(clusters_query)
        clusters = ",".join([str(cluster[0]) for cluster in clusters])

        query = "DELETE FROM %s WHERE storage_id = %s AND cluster_id in (%s);" % (self._name, self.storage_id, clusters)
        query2 = "DELETE FROM hecuba.istorage WHERE storage_id = %s;" % self.storage_id
        log.debug("DELETE PERSISTENT: %s", query)
        config.session.execute(query)
        config.session.execute(query2)
        self.storage_id = None

    def __iter__(self):
        if self._numpy_full_loaded:
            return iter(self.view(np.ndarray))
        else:
            return iter(self[:].view(np.ndarray))

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
        if self._is_persistent and len(self.shape) and self._numpy_full_loaded is False:
            self._hcache.load_numpy_slices([self.storage_id], self._build_args.metas, [self.base.view(np.ndarray)],
                                           None)

        results = super(StorageNumpy, self).__array_ufunc__(ufunc, method,
                                                            *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            return

        if self._is_persistent and len(self.shape):
            self._hcache.store_numpy_slices([self.storage_id], self._build_args.metas, [self.base.view(np.ndarray)],
                                            None)

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((result
                         if output is None else output)
                        for result, output in zip(results, outputs))

        return results[0] if len(results) == 1 else results
