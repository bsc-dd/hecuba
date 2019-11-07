import itertools as it
from collections import namedtuple

import numpy as np
from hfetch import HNumpyStore

from . import config, log
from .IStorage import IStorage
from .tools import extract_ks_tab, get_istorage_attrs


class StorageNumpy(IStorage, np.ndarray):
    class np_meta(object):
        def __init__(self, shape, dtype, block_id):
            self.dims = shape
            self.type = dtype
            self.block_id = block_id

    _build_args = None
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba.istorage'
                                                  '(storage_id, class_name, name, numpy_meta)'
                                                  'VALUES (?,?,?,?)')

    args_names = ["storage_id", "class_name", "name", "shape", "dtype", "block_id", "built_remotely"]
    args = namedtuple('StorageNumpyArgs', args_names)

    def __new__(cls, input_array=None, storage_id=None, name=None, built_remotely=False, **kwargs):
        if storage_id and not name:
            metas = get_istorage_attrs(storage_id)
            name = metas[0].name
        elif not name:
            name = ''
        if input_array is None and name and storage_id is not None:
            result = cls.reserve_numpy_array(storage_id, name)
            input_array = result[0]
            obj = np.asarray(input_array).view(cls)
            obj._hcache = result[1]
        elif not name and storage_id is not None:
            raise RuntimeError("hnumpy received storage id but not a name")
        else:
            obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        obj._set_name(name)
        obj._numpy_full_loaded = False
        obj._loaded_coordinates = []
        obj._block_id = -1
        obj._built_remotely = built_remotely
        obj._class_name = '%s.%s' % (cls.__module__, cls.__name__)
        return obj

    def __init__(self, input_array=None, storage_id=None, name=None, **kwargs):
        IStorage.__init__(self, storage_id=storage_id, name=self._get_name(), **kwargs)
        if self._get_name() or self.storage_id:
            if input_array is not None:
                self.make_persistent(self._get_name())
            self._row_elem = self._hcache.get_elements_per_row(self.storage_id)[0]
            self._is_persistent = True

    # used as copy constructor
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.storage_id = getattr(obj, 'storage_id', None)
        self.name = getattr(obj, 'name', None)
        self._hcache = getattr(obj, '_hcache', None)
        self._row_elem = getattr(obj, '_row_elem', None)
        self._loaded_coordinates = getattr(obj, '_loaded_coordinates', None)
        self._numpy_full_loaded = getattr(obj, '_numpy_full_loaded', None)
        self._is_persistent = getattr(obj, '_is_persistent', None)

    @staticmethod
    def _create_tables(name):
        (ksp, _) = extract_ks_tab(name)
        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (ksp, config.replication)
        config.session.execute(query_keyspace)

        config.session.execute(
            'CREATE TABLE IF NOT EXISTS ' + name + '(storage_id uuid , '
                                                                'cluster_id int, '
                                                                'block_id int, '
                                                                'payload blob, '
                                                                'PRIMARY KEY((storage_id,cluster_id),block_id))')

    @staticmethod
    def _create_hcache(name):
        (ksp, name) = extract_ks_tab(name)
        hcache_params = (ksp, name,
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
                                    storage_args.name, StorageNumpy.np_meta(storage_args.shape, storage_args.dtype,
                                                                            storage_args.block_id)])

        except Exception as ex:
            log.warn("Error creating the StorageNumpy metadata with args: %s" % str(storage_args))
            raise ex

    @staticmethod
    def reserve_numpy_array(storage_id, name):
        '''Provides a numpy array with the number of elements obtained through storage_id'''

        hcache = StorageNumpy._create_hcache(name)
        result = hcache.allocate_numpy(storage_id)
        if len(result) == 1:
            return [result[0], hcache]
        else:
            raise KeyError

    def generate_coordinates(self, coordinates):
        coords = None
        if coordinates is None:
            coords = []
        elif len(coordinates) == 1:
            coord = coordinates // self._row_elem
            coords = [(c,) for c in range(np.min(coord), np.max(coord) + 1)]
        else:
            # coords divided by number of elem in a row
            coord = [coordinates[:, coord] // self._row_elem for coord in range(len(coordinates[0]))]
            ranges = (range(*range_tuple) for range_tuple in zip(coord[0], coord[1] + 1))
            coords = list(it.product(*ranges))
        return coords

    def format_coords(self, coord):
        if not isinstance(coord, tuple):
            coord = (coord,)

        if not all(isinstance(c, slice) for c in coord):
            raise IndexError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices")

        np_list = []
        count_none = 0
        for dim, coo in enumerate(coord):
            if coo.stop is None and coo.start is None:
                np_list.append([0, self.shape[dim]])
                count_none = count_none + 1
            elif coo.start is None:
                np_list.append([0, coo.stop])
            elif coo.stop is None:
                np_list.append([coo.start, self.shape[dim]])
            else:
                np_list.append([coo.start, coo.stop])

        if count_none == len(coord): return None
        coordinates = np.array(np_list)
        return coordinates

    def slices_match_numpy_shape(self, sliced_coord):
        if sliced_coord is None:
            return True
        elif len(self.shape) != len(sliced_coord):
            return False
        else:
            for i, queried_slice in enumerate(sliced_coord):
                match_shape = queried_slice[1] <= self.shape[i]
        return match_shape

    def get_coords_match_numpy_shape(self, coo):
        new_coords = self.generate_coordinates(coo)
        shape_slices = tuple([slice(0, coord) for coord in self.shape])
        formated_shape_coords = self.format_coords(shape_slices)
        numpy_coords = self.generate_coordinates(formated_shape_coords)
        coordinates = []
        for coord in new_coords:
            if coord in numpy_coords and coord not in coordinates:
                coordinates.append(coord)
        return coordinates

    def __getitem__(self, sliced_coord):
        log.info("RETRIEVING NUMPY")
        if self._is_persistent and not (self._loaded_coordinates or self._numpy_full_loaded):
            # formats sliced coords
            new_coords = self.format_coords(sliced_coord)

            # checks if some coord in sliced_coords are inside the numpy
            if not self.slices_match_numpy_shape(new_coords):  # some coordinates match
                new_coords = self.get_coords_match_numpy_shape(new_coords)  # generates the coordinates
                if not new_coords:
                    self._hcache.load_numpy_slices([self.storage_id], [self.view(np.ndarray)],
                                                   None)  # any coordinates generated match
                    self._numpy_full_loaded = True
                    return super(StorageNumpy, self).__getitem__(sliced_coord)
            else:  # coordinates match
                new_coords = self.generate_coordinates(new_coords)

            # coordinates is the union between the loaded coordiantes and the new ones
            coordinates = list(set(it.chain.from_iterable((self._loaded_coordinates, new_coords))))

            # checks if we already loaded the coordinates
            if ((len(coordinates) != len(self._loaded_coordinates)) and not self._numpy_full_loaded) or (
                    not self._numpy_full_loaded and not coordinates):
                if not coordinates:
                    self._numpy_full_loaded = True
                    new_coords = None
                self._hcache.load_numpy_slices([self.storage_id], [self.view(np.ndarray)], new_coords)
                self._loaded_coordinates = coordinates
        return super(StorageNumpy, self).__getitem__(sliced_coord)

    def __setitem__(self, sliced_coord, values):
        log.info("WRITING NUMPY")
        if self._is_persistent:
            coo = self.format_coords(sliced_coord)
            coordinates = list(set(it.chain.from_iterable(
                (self._loaded_coordinates, self.generate_coordinates(coo)))))
            self._hcache.store_numpy_slices([self.storage_id], [self.view(np.ndarray)], coordinates)
        return super(StorageNumpy, self).__setitem__(sliced_coord, values)

    def make_persistent(self, name):
        super().make_persistent(name)

        self._build_args = self.args(self.storage_id, self._class_name, name,
                                     self.shape, self.dtype.num, self._block_id, self._built_remotely)

        if not self._built_remotely:
            self._create_tables(name)

        if not getattr(self, '_hcache', None):
            self._hcache = self._create_hcache(name)

        if len(self.shape) != 0:
            self._hcache.store_numpy_slices([self.storage_id], [self.view(np.ndarray)], None)
        StorageNumpy._store_meta(self._build_args)

    def stop_persistent(self):
        super().stop_persistent()

        self.storage_id = None

    def delete_persistent(self):
        """
            Deletes the Cassandra table where the persistent StorageObj stores data
        """
        super().delete_persistent()
        query = "DELETE FROM {} WHERE storage_id = {} AND cluster_id=-1;".format(self._get_name(),
                                                                                    self.storage_id)
        query2 = "DELETE FROM hecuba.istorage WHERE storage_id = %s;" % self.storage_id
        log.debug("DELETE PERSISTENT: %s", query)
        config.session.execute(query)
        config.session.execute(query2)

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
        if self._is_persistent and len(self.shape) and self._numpy_full_loaded is False:
            self._hcache.load_numpy_slices([self.storage_id], [self.view(np.ndarray)], None)

        results = super(StorageNumpy, self).__array_ufunc__(ufunc, method,
                                                            *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            return

        if self._is_persistent and len(self.shape):
            self._hcache.store_numpy_slices([self.storage_id], [self.view(np.ndarray)], None)

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((result
                         if output is None else output)
                        for result, output in zip(results, outputs))

        return results[0] if len(results) == 1 else results
