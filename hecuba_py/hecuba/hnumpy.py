import itertools as it
import uuid
from collections import namedtuple
from typing import Tuple

from math import ceil

import numpy as np
from hfetch import HNumpyStore, HArrayMetadata

from . import config, log
from .IStorage import IStorage
from .tools import extract_ks_tab, get_istorage_attrs, storage_id_from_name


class StorageNumpy(IStorage, np.ndarray):
    _build_args = None

    # twin_id - the storage_id to the 'arrow' companion of this numpy
    #    (A Numpy has two copies: 1) with rows and 2) with columns (arrow))
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba.istorage'
                                                  '(storage_id, class_name, name, numpy_meta, block_id, base_numpy, twin_id)'
                                                  'VALUES (?,?,?,?,?,?,?)')

    args_names = ["storage_id", "class_name", "name", "metas", "block_id", "base_numpy", "twin_id"]
    args = namedtuple('StorageNumpyArgs', args_names)

    def _calculate_coords(self):
        ''' Calculate all the coordinates given the shape of a numpy array to avoid calculate them at each get&set
            This is used to determine if a numpy is full loaded on memory and avoid accesses to cassandra.
        '''
        self._all_coords = np.array(list(np.ndindex(self.shape))).reshape(*self.shape,self.ndim)
        num_blocks = 1
        for i in range(0, self.ndim):
            b = ceil(self.shape[i] / self._row_elem)
            num_blocks = num_blocks * b
        self._n_blocks = num_blocks

    def np_split(self, block_size: Tuple[int, int]):
        # For now, only split in two dimensions is supported
        bn, bm = block_size
        for block_id, i in enumerate(range(0, self.shape[0], bn)):
            block = [self[i: i + bn, j:j + bm] for j in range(0, self.shape[1], bm)]
            obj = StorageNumpy(input_array=block, name=self._get_name(), storage_id=uuid.uuid4(), block_id=block_id)
            yield obj

    @staticmethod
    def get_arrow_name(ksp, name):
        # get_arrow_name: Returns the name of the arrow table (READ) of a table_name
        return ksp + "_arrow." + name +"_arrow"

    @staticmethod
    def _isarrow(name):
        '''
        Returns true if the name is an arrow table
        '''
        return name.endswith("_arrow")

    @staticmethod
    def get_buffer_name(ksp, name):
        """
        Returns a full qualified name for a table name in the arrow keyspace
        """
        return ksp+"_arrow." + name +"_buffer"

    # _initialize_existing_object : Instantiates a new StorageNumpy
    # from metadata existent in Hecuba given its name or storage_id.
    #   Parameters:
    #       cls       : Class to use for instantiation
    #       name      : A *qualified* cassandra name (keyspace.table_name) to instantiate
    #       storage_id: The UUID to instantiate
    # If both, name and storage_id, are given, name is ignored.
    # It reserves memory to store the numpy (zeros).
    @staticmethod
    def _initialize_existing_object(cls, name, storage_id):
        #    StorageNumpy(None, name="xxx", none)
        # or StorageNumpy(None, none, storage_id="xxx")
        # or StorageNumpy(None, name="xxx", storage_id="yyy") Does it exist?
        log.debug("name=%s sid=%s", name, storage_id)
        if not storage_id:  # StorageNumpy(None, name="xxxx", NONE)
            #(ksp, table) = extract_ks_tab(name)
            #name = ksp + "." + table
            storage_id = storage_id_from_name(name)

        # Load metadata
        istorage_metas = get_istorage_attrs(storage_id)
        name = istorage_metas[0].name
        twin_id = istorage_metas[0].twin_id # This MUST match 'storage_id_from_name(_twin_name)'
        numpy_metadata = istorage_metas[0].numpy_meta
        base_metas = numpy_metadata
        base_numpy = istorage_metas[0].base_numpy
        if base_numpy is not None:
            storage_id = base_numpy
            base_metas = get_istorage_attrs(storage_id)[0].numpy_meta

        # Load array
        result = cls.reserve_numpy_array(storage_id, name, base_metas)
        input_array = result[0]

        obj = np.asarray(input_array).view(cls)

        obj._hcache = result[1]
        (obj._ksp, obj._table) = extract_ks_tab(name)
        obj._set_name(name)
        obj.storage_id = storage_id
        if twin_id is not None:
            # Load TWIN array
            #print ("JJ __new__ twin_name ", obj._twin_name, flush=True);
            #print ("JJ __new__ twin_id ", obj._twin_id, flush=True);
            obj._twin_id   = twin_id
            obj._twin_name = StorageNumpy.get_arrow_name(obj._ksp, obj._table)
            twin = StorageNumpy._initialize_existing_object(cls, obj._twin_name, obj._twin_id)
            obj._twin_ref = twin
        obj._row_elem = obj._hcache.get_elements_per_row(storage_id, base_metas)
        if base_numpy is not None:
            obj._partition_dims = numpy_metadata.dims
        obj._is_persistent = True
        obj._calculate_coords()
        #print (" JJ _initialize_existing_object name={} sid={} DONE".format(name, storage_id), flush=True)
        return obj

    # _complete_initialization: Complete the fields of a new initialized object
    # To be called as the last step of an StorageNumpy object creation
    def _complete_initialization(self, cls, name, block_id):
        self._numpy_full_loaded = False
        self._loaded_coordinates = []
        self._set_name(name)
        if getattr(self, "_block_id", None) is None:
            self._block_id = block_id
        # Finally, we must return the newly created object:
        self._class_name = '%s.%s' % (cls.__module__, cls.__name__)
        if self._twin_ref is not None:
            self._twin_ref._complete_initialization(cls, self._twin_name, block_id)

    def __new__(cls, input_array=None, name=None, storage_id=None, block_id=None, **kwargs):
        log.debug("input_array=%s name=%s storage_id=%s ",input_array is not None, name, storage_id)
        if name is not None:
            # Construct full qualified name to deal with cases where the name does NOT contain keyspace
            (ksp, table) = extract_ks_tab(name)
            name = ksp + "." + table
            if (len(table)>40 or table.startswith("HECUBA")):
                # Cassandra limits name to 48 characters: we reserve 8 characters
                # for special Hecuba tables
                raise AttributeError("The name of an user StorageNumpy is limited to 40 chars and can not start 'HECUBA'")

        if input_array is None and (name is not None or storage_id is not None):
            obj = StorageNumpy._initialize_existing_object(cls, name, storage_id)
            name = obj._get_name()

        else:
            if isinstance(input_array, StorageNumpy): # StorageNumpyDesign
                # StorageNumpy(Snumpy, None, None)
                # StorageNumpy(Snumpy, name, None)
                # StorageNumpy(Snumpy, None, UUID)
                # StorageNumpy(Snumpy, name, UUID)
                if storage_id is not None:
                    log.warn("Creating a StorageNumpy with a specific StorageID")
                if not input_array._is_persistent and name is None:
                    raise NotImplemented("Create a volatile StorageNumpy from another volatile StorageNumpy")

                obj = input_array.view(cls)
                if input_array._is_persistent:
                    sid=uuid.uuid4()
                    obj.storage_id=sid
                    if name is not None:
                        raise NotImplemented("Build a Persistent StorageNumpy from another Persistent StorageNumpy with a name")
                    name = input_array._get_name()

                if input_array._twin_ref is not None:
                    # FIXME Treat the TWIN also!
                    # Check that the new object has the same ref as base
                    log.warn("Creating a volatile StorageNumpy. Ignoring TWIN")
                    pass
            else:
                # StorageNumpy(numpy, None, None)
                obj = np.asarray(input_array).copy().view(cls)
                if getattr(input_array, 'ndim', 0) == 2:
                    obj._twin_id  = None
                    obj._twin_name = None
                    obj._twin_ref = np.asarray(input_array).T.copy().view(cls)
                    log.debug("Created TWIN")

        #print("JJ name = ", name, flush=True)
        #print("JJ _twin_name = ", obj._twin_name, flush=True)
        #print("JJ _name = ", obj._name, flush=True)
        obj._complete_initialization(cls, name, block_id)
        return obj

    def __init__(self, input_array=None, name=None, storage_id=None, **kwargs):
        log.debug("name=%s sid=%s", name, storage_id)
        #if we already have a name we keep it and ignore the new one
        current_name=self._get_name()
        if current_name is not None:
            name=current_name
        # was to ovewrite the former name with the new one
        #if name == None:
        #    name=self._get_name() # To deal with StorageNumpy(StorageNumpy)

        IStorage.__init__(self, storage_id=storage_id, name=name, **kwargs)
        metas = HArrayMetadata(list(self.shape), list(self.strides), self.dtype.kind, self.dtype.byteorder,
                               self.itemsize, self.flags.num, 0)
        self._build_args = self.args(self.storage_id, self._class_name,
                self._get_name(), metas, self._block_id, self.storage_id, self._twin_id)
        twin = self._twin_ref
        if twin is not None:
            t_name=getattr(twin, 'name', None)
            IStorage.__init__(twin, storage_id=twin.storage_id, name=t_name, **kwargs)
            twin_metas = HArrayMetadata(
                                        list(twin.shape),
                                        list(twin.strides),
                                        twin.dtype.kind,
                                        twin.dtype.byteorder,
                                        twin.itemsize,
                                        twin.flags.num,
                                        2)  # 2 == COLUMNAR (find it at SpaceFillingCurve.h)
            twin._build_args = twin.args(
                                         self._twin_id,
                                         self._class_name,
                                         self._twin_name,
                                         twin_metas,
                                         None, #self._block_id,
                                         self._twin_id, # base numpy
                                         None) #twin_id

        if self._get_name() or self.storage_id:
            load_data= (input_array is None) and (config.load_on_demand == False)
            if input_array is not None:
                if isinstance(input_array,StorageNumpy):
                    self._build_args = self.args(self.storage_id,
                                                 self._class_name,
                                                 self._get_name(),
                                                 metas,
                                                 self._block_id,
                                                 input_array._build_args.base_numpy, # Update base_numpy with original_data
                                                 self._twin_id)
                    self._store_meta(self._build_args) # StoreNumpy from persistent StorageNumpy (reshape): we only need to update the metadata
                    # FIXME twin?
                    log.warn("Creating a Persistent StorageNumpy. Ignoring TWIN")
                else:
                    self.make_persistent(self._get_name())
            self._is_persistent = True # IStorage.__init__ resets this! Therefore enable it again
            if load_data:
                self[:]	# HACK! Load ALL elements in memory NOW (recursively calls getitem)


    # used as copy constructor
    def __array_finalize__(self, obj):
        if obj is None:
            log.debug("  __array_finalize__ NEW")
            return
        log.debug("__array_finalize__ self.base=None?%s obj.base=None?%s", getattr(self, 'base', None) is None, getattr(obj, 'base', None) is None)
        if self.base is not None: # It is a view, therefore, copy data from object
            log.debug("  __array_finalize__ view (new_from_template/view)")

            self.storage_id = getattr(obj, 'storage_id', None)
            self._name = getattr(obj, '_name', None)
            self._hcache = getattr(obj, '_hcache', None)
            self._row_elem = getattr(obj, '_row_elem', None)
            # if we are a view we have ALREADY loaded all the subarray
            self._loaded_coordinates = getattr(obj, '_loaded_coordinates', None)
            self._numpy_full_loaded = getattr(obj, '_numpy_full_loaded', None)
            self._is_persistent = getattr(obj, '_is_persistent', None)
            self._block_id = getattr(obj, '_block_id', None)
            self._twin_id   = getattr(obj, '_twin_id', None)
            self._twin_ref  = getattr(obj, '_twin_ref', None)
            self._twin_name = getattr(obj, '_twin_name', None)
            self._all_coords = getattr(obj, '_all_coords', None)
            self._n_blocks = getattr(obj, '_n_blocks', None)
            self._class_name = getattr(obj,'_class_name',None)
            hfetch_metas = HArrayMetadata(list(self.shape), list(self.strides), self.dtype.kind, self.dtype.byteorder,
                                          self.itemsize, self.flags.num, 0)
            self._build_args = getattr(obj, '_build_args', self.args(self.storage_id, self._class_name, self._get_name(), hfetch_metas,
                                        self._block_id, None, self._twin_id))
        else:
            log.debug("  __array_finalize__ copy")
            # Initialize fields as the __new__ case with input_array and not name
            self._loaded_coordinates = []
            self._numpy_full_loaded  = False
            self._name               = None
            self.storage_id          = None
            self._is_persistent      = False


    @staticmethod
    def _get_base_array(self):
        ''' Returns the 'base' numpy from this SN.  '''
        return (getattr(self,'base',self))

    @staticmethod
    def _create_tables(name):
        (ksp, table) = extract_ks_tab(name)
        log.debug("Create table %s %s", ksp, table)
        if StorageNumpy._isarrow(table): # Arrow tables are already created, skip them
                return
        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (ksp, config.replication)
        config.executelocked(query_keyspace)


        query_table='CREATE TABLE IF NOT EXISTS ' + ksp + '.' + table + '(storage_id uuid , '\
                                                                'cluster_id int, '           \
                                                                'block_id int, '             \
                                                                'payload blob, '             \
                                                                'PRIMARY KEY((storage_id,cluster_id),block_id))'
        config.executelocked(query_table)

        # Add 'arrow' tables
        #	harrow_ to read
        #	buffer_ to write
        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (ksp+"_arrow", config.replication)
        config.executelocked(query_keyspace)

        tbl_buffer = StorageNumpy.get_buffer_name(ksp, table)

        query_table_buff ='CREATE TABLE IF NOT EXISTS ' + tbl_buffer + \
                                                                '(storage_id uuid , '    \
                                                                'col_id      bigint, '   \
                                                                'row_id      bigint, '   \
                                                                'size_elem   int, '      \
                                                                'payload     blob, '     \
                                                                'PRIMARY KEY(storage_id,col_id))'
        config.executelocked(query_table_buff)
        tbl_arrow = StorageNumpy.get_arrow_name(ksp, table)
        query_table_arrow='CREATE TABLE IF NOT EXISTS ' + tbl_arrow + \
                                                                '(storage_id uuid, '    \
                                                                'col_id      bigint, '  \
                                                                'arrow_addr  bigint, '  \
                                                                'arrow_size  int, '     \
                                                                'PRIMARY KEY(storage_id,col_id))'
        config.executelocked(query_table_arrow)

    @staticmethod
    def _create_hcache(name):
        (ksp, table) = extract_ks_tab(name)
        log.debug("Create cache for %s %s", ksp, table)
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
                                    storage_args.name,
                                    storage_args.metas,
                                    storage_args.block_id,
                                    storage_args.base_numpy,
                                    storage_args.twin_id])

        except Exception as ex:
            log.warn("Error creating the StorageNumpy metadata with args: %s" % str(storage_args))
            raise ex

    @staticmethod
    def reserve_numpy_array(storage_id, name, metas):
        '''Provides a numpy array with the number of elements obtained through storage_id'''
        log.debug(" Reserve memory for %s %s ", name, storage_id)
        hcache = StorageNumpy._create_hcache(name)
        result = hcache.allocate_numpy(storage_id, metas)
        if len(result) == 1:
            return [result[0], hcache]
        else:
            raise KeyError

    def _select_blocks(self,sliced_coord):
        """
            Calculate the list of block coordinates to load given a specific numpy slice syntax.
            Args:
                self: StorageNumpy to apply list of coordinates.
            Calculate the list of block coordinates to load given a specific numpy slice syntax.
            Args:
                self: StorageNumpy to apply list of coordinates.
                sliced_coord: Slice syntax to evaluate.
            May raise an IndexError exception
        """
        if isinstance(sliced_coord, slice) and sliced_coord == slice(None, None, None):
            new_coords = None
        else:
            new_coords = self._all_coords[sliced_coord] // self._row_elem
            new_coords = new_coords.reshape(-1, self.ndim)
            new_coords = [tuple(coord) for coord in new_coords]
            new_coords = list(dict.fromkeys(new_coords))

        log.debug("selecting blocks :{} -> {}".format(sliced_coord,new_coords))
        return new_coords


    def _load_blocks(self, new_coords):
        """
            Load the provided block coordinates from cassandra into memory
            Args:
                self: The StorageNumpy to load data into
                new_coords: The coordinates to load (using ZOrder identification)
        """
        if self._is_persistent is False:
            raise NotImplemented("NOT SUPPORTED to load coordinates on a volatile object")

        if not new_coords: # Special case: Load everything
            log.debug("LOADING ALL BLOCKS OF NUMPY")
            self._numpy_full_loaded = True
            new_coords = None
            base_numpy = StorageNumpy._get_base_array(self)
            self._hcache.load_numpy_slices([self._build_args.base_numpy], self._build_args.metas, [base_numpy],
                                       new_coords)
            self._loaded_coordinates = None
        else:
            log.debug("LOADING COORDINATES")

            # coordinates is the union between the loaded coordinates and the new ones
            coordinates = list(set(it.chain.from_iterable((self._loaded_coordinates, new_coords))))
            if (len(coordinates) != len(self._loaded_coordinates)):
                self._numpy_full_loaded = (len(coordinates) == self._n_blocks)

                base_numpy = StorageNumpy._get_base_array(self)
                self._hcache.load_numpy_slices([self._build_args.base_numpy], self._build_args.metas, [base_numpy],
                                       new_coords)
                self._loaded_coordinates = coordinates

    def _select_columns(self, sliced_coord):
        """
        Returns None or a list of columns accesed
        """
        columns = None
        if isinstance(sliced_coord, tuple):
            # If the getitem parameter is a tuple, then we may catch the
            # column accesses: Ex: s[:, i], s[:, [i1,i2]], s[:, slice(...)]
            # All these accesses arrive here as a tuple:
            #   (slice(None,None,None), xxx)
            # where xxx is the last parameter of the tuple.
            # FIXME Extend to more than 2 dimensions
            dims = sliced_coord.__len__()
            if dims == 2: # Only 2 dimensions
                if isinstance(sliced_coord[-dims], slice) and sliced_coord[-dims] == slice(None, None, None):
                    # A WHOLE COLUMN selected!
                    if isinstance(sliced_coord[-1], slice):
                        log.warn("Columnar access and slice is not implemented")
                        return None
                    columns = []
                    columns.append(sliced_coord[-1])
        return columns

    def _load_columns(self, columns):
        """
            Load from Cassandra the list of columns.
            This accesses the twin.
            Args:
                self: The StorageNumpy to load data into
                columns: The coordinates to load (column position)
        """

        if self._twin_ref._is_persistent:
            log.debug("LOADING COLUMNS {}".format(columns))
            base_numpy = StorageNumpy._get_base_array(self._twin_ref)
            self._twin_ref._hcache.load_numpy_slices([self._twin_ref._build_args.base_numpy],
                                        self._twin_ref._build_args.metas,
                                        [base_numpy],
                                        columns)

    def _select_and_load_blocks(self, sliced_coord):
        if self._is_persistent:
            if not self._numpy_full_loaded:
                block_coord = self._select_blocks(sliced_coord)
                self._load_blocks(block_coord)

    def __getitem__(self, sliced_coord):
        log.info("RETRIEVING NUMPY {}".format(sliced_coord))
        if self._is_persistent:
            #if the slice is a npndarray numpy creates a copy and we do the same
            if isinstance(sliced_coord, np.ndarray): # is there any other slicing case that needs a copy of the array????
                result = self.view(np.ndarray)[sliced_coord]
                return StorageNumpy(result) # Creates a copy (A StorageNumpy from a Numpy)

            if self._build_args.metas.partition_type == 2 :
                #HACK if the accessed numpy is a columnar one, assume that it is already in memory
                log.warn("Accessing a twin directly. Assuming it is already in memory")
                return super(StorageNumpy, self).__getitem__(sliced_coord)

        # Check if the access is columnar...
        columns = self._select_columns(sliced_coord)
        if columns is not None :
            self._load_columns(columns)
            return super(StorageNumpy,
                    self._twin_ref).__getitem__((columns[0], slice(None, None, None)))

        # Normal array access...
        self._select_and_load_blocks(sliced_coord)
        return super(StorageNumpy, self).__getitem__(sliced_coord)

    def __setitem__(self, sliced_coord, values):
        log.info("WRITING NUMPY")
        log.debug("setitem %s", sliced_coord)
        if self._is_persistent:
            block_coords = self._select_blocks(sliced_coord)
            if not self._numpy_full_loaded: # Load the block before writing!
                self._load_blocks(block_coords)

            #yolandab: execute first the super to modified the base numpy

            modified_np=super(StorageNumpy, self).__setitem__(sliced_coord, values)

            base_numpy = StorageNumpy._get_base_array(self) # self.base is  numpy.ndarray
            self._hcache.store_numpy_slices([self._build_args.base_numpy],
                    self._build_args.metas, [base_numpy],
                    block_coords)
            #if self._twin_ref is not None:
            #    super(StorageNumpy, self._twin_ref).__setitem__(sliced_coord, values)
            #    self._twin_ref._hcache.store_numpy_slices([self._twin_ref._build_args.base_numpy],
            #            self._twin_ref._build_args.metas,
            #            [self._twin_ref.base.view(np.ndarray)],
            #            sliced_coord[-1])
            return modified_np
        #if self._twin_ref is not None:
        #    super(StorageNumpy, self._twin_ref).__setitem__(sliced_coord, values)
        return super(StorageNumpy, self).__setitem__(sliced_coord, values)

    def make_persistent(self, name):
        log.debug("Make %s persistent", name)

        super().make_persistent(name)

        if not self._built_remotely:
            self._create_tables(name)

        if not getattr(self, '_hcache', None):
            self._hcache = self._create_hcache(name)

        if None in self or not self.ndim:
            raise NotImplemented("Empty array persistance")

        twin = self._twin_ref
        if twin is not None :
            # If there is a twin, make it persistent FIRST
            twksp, twtbl = extract_ks_tab(name)
            twinname = StorageNumpy.get_arrow_name(twksp, twtbl)
            # 1) make it persistent
            super(StorageNumpy, twin).make_persistent(twinname)
            self._twin_id = twin.storage_id
            self._twin_name = twin._get_name()

            if not getattr(twin, '_hcache', None):
                twin._hcache = twin._create_hcache(twinname)

            # 2) Build metadatas for Twin with different metadata
            twin_metas = HArrayMetadata(
                                        list(twin.shape),
                                        list(twin.strides),
                                        twin.dtype.kind,
                                        twin.dtype.byteorder,
                                        twin.itemsize,
                                        twin.flags.num,
                                        2)  # 2 == COLUMNAR (find it at SpaceFillingCurve.h)
            twin._build_args = twin.args(
                                         self._twin_id,
                                         self._class_name,
                                         self._twin_name,
                                         twin_metas,
                                         None, #self._block_id,
                                         self._twin_id, # base numpy
                                         None) #twin_id
            # 3) Store metadata
            StorageNumpy._store_meta(twin._build_args)

            # 4) Store the data in cassandra
            if len(twin.shape) != 0:
                twin._hcache.store_numpy_slices([self._twin_ref._build_args.base_numpy], twin._build_args.metas, [twin.base.view(np.ndarray)],
                                                None)

        # Persist current object
        hfetch_metas = HArrayMetadata(list(self.shape), list(self.strides), self.dtype.kind, self.dtype.byteorder,
                                      self.itemsize, self.flags.num, 0)# 0 ==Z_ORDER (find it at SpaceFillingCurve.h)
        self._build_args = self.args(self.storage_id, self._class_name, self._get_name(), hfetch_metas, self._block_id,
                                     self.storage_id, # base_numpy is storage_id because until now we only reach this point if we are not inheriting from a StorageNumpy. We should update this if we allow StorageNumpy from volatile StorageNumpy
                                     self._twin_id)
        if len(self.shape) != 0:
            self._hcache.store_numpy_slices([self._build_args.base_numpy], self._build_args.metas, [StorageNumpy._get_base_array(self)],
                                            None)
        StorageNumpy._store_meta(self._build_args)
        self._row_elem = self._hcache.get_elements_per_row(self.storage_id, self._build_args.metas)
        self._calculate_coords()



    def stop_persistent(self):
        super().stop_persistent()

        self.storage_id = None

    def delete_persistent(self):
        """
            Deletes the Cassandra table where the persistent StorageObj stores data
        """
        super().delete_persistent()

        clusters_query = "SELECT cluster_id FROM %s WHERE storage_id = %s ALLOW FILTERING;" % (
            self._get_name(), self.storage_id)
        clusters = config.session.execute(clusters_query)
        clusters = ",".join([str(cluster[0]) for cluster in clusters])

        query = "DELETE FROM %s WHERE storage_id = %s AND cluster_id in (%s);" % (self._get_name(), self.storage_id, clusters)
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

        base_numpy = StorageNumpy._get_base_array(self)
        if self._is_persistent and len(self.shape) and self._numpy_full_loaded is False:
            self._hcache.load_numpy_slices([self._build_args.base_numpy], self._build_args.metas, [base_numpy],
                                           None)
        results = super(StorageNumpy, self).__array_ufunc__(ufunc, method,
                                                            *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            return

        if self._is_persistent and len(self.shape):
            self._hcache.store_numpy_slices([self._build_args.base_numpy], self._build_args.metas, [base_numpy],
                                            None)

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((result
                         if output is None else output)
                        for result, output in zip(results, outputs))

        return results[0] if len(results) == 1 else results


    def reshape(self, newshape, order="C"):
        '''
        reshape the StorageNumpy
        Creates a view of the StorageNumpy sharing data with the original data (Both in disk and memory)
        '''
        log.debug("reshape from %s to %s", self.shape, newshape)
        obj=super(StorageNumpy, self).reshape(newshape, order)
        return obj

    def transpose(self,axes=None):
        '''
        transpose the StorageNumpy
        Creates a view of the StorageNumpy sharing data with the original data (Both in disk and memory)
        '''
        obj=super(StorageNumpy, self).transpose(axes)
        return obj

    def copy(self, order='K'):
        '''
        Copy a StorageNumpy: new **volatile** StorageNumpy with the data of the parameter
        '''
        n_sn=super(StorageNumpy,self).copy(order)
        if self._twin_ref is not None:
            n_sn._twin_id = None
            n_sn._twin_name = None
            n_sn._twin_ref = super(StorageNumpy, self._twin_ref).copy(order)
        return n_sn


    ###### INTERCEPTED FUNCTIONS #####
    def dot(a, b, out=None):
        if isinstance(a, StorageNumpy) and a._is_persistent and not a._numpy_full_loaded:
           a[:]	# HACK! Load ALL elements in memory NOW (recursively calls getitem)

        if isinstance(b, StorageNumpy) and b._is_persistent and not b._numpy_full_loaded:
           b[:]	# HACK! Load ALL elements in memory NOW (recursively calls getitem)

        return config.intercepted['dot'](a,b,out)
