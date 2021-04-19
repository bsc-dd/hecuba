import itertools as it
import uuid
from collections import namedtuple
from typing import Tuple

from math import ceil

import numpy as np
from hfetch import HNumpyStore, HArrayMetadata

from . import config, log
from .IStorage import IStorage
from .tools import extract_ks_tab, get_istorage_attrs, storage_id_from_name, build_remotely


class StorageNumpy(IStorage, np.ndarray):
    _build_args = None

    # twin_id - the storage_id to the 'arrow' companion of this numpy
    #    (A Numpy has two copies: 1) with rows and 2) with columns (arrow))
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba.istorage'
                                                  '(storage_id, class_name, name, numpy_meta, block_id, base_numpy, twin_id, tokens)'
                                                  'VALUES (?,?,?,?,?,?,?,?)')

    args_names = ["storage_id", "class_name", "name", "metas", "block_id", "base_numpy", "twin_id", "tokens"]
    args = namedtuple('StorageNumpyArgs', args_names)

    def _calculate_coords(self, metas):
        ''' Calculate all the coordinates given the shape of a numpy array to avoid calculate them at each get&set
            This is used to determine if a numpy is full loaded on memory and avoid accesses to cassandra.
            Args:
                self : object to add new coords
                metas: metadatas to use to calculate the coordenates (may be different from the self.metas)
        '''

        ndim = len(metas.dims)
        self._all_coords = np.array(list(np.ndindex(tuple(metas.dims)))).reshape(*tuple(metas.dims), ndim)
        num_blocks = 1
        for i in range(0, ndim):
            b = ceil(metas.dims[i] / self._row_elem)
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
    def _composite_key(storage_id, cluster_id):
        """
        Calculate the cassandra hash (storage_id, cluster_id) courtesy of:
        https://stackoverflow.com/questions/22915237/how-to-generate-cassandra-token-for-composite-partition-key
        """
        from cassandra import murmur3
        tam_sid = len(storage_id.bytes)
        bytes_storage_id = bytes([0, tam_sid]) + storage_id.bytes + bytes([0])
        bytes_cluster_id = bytes([0, 4]) + bytes([0, 0, 0, cluster_id]) + bytes([0])
        mykey = bytes_storage_id + bytes_cluster_id
        return murmur3.murmur3(mykey)


    def split(self):
        # TODO this should work for VOLATILE objects too! Now only works for PERSISTENT
        if self._build_args.metas.partition_type == 2:
            raise NotImplementedError("Split on columnar data is not supported")

        blocks = self._hcache.get_block_ids(self._build_args.metas) # returns a list of tuples (cluster_id, block_id)

        #Build map cluster_id -> token
        cluster_id_to_token = {}
        for (zorder_id, cluster_id, block_id, ccs) in blocks:
            if not cluster_id_to_token.__contains__(cluster_id):
                hash_key = StorageNumpy._composite_key(self.storage_id, cluster_id)
                cluster_id_to_token[cluster_id] = hash_key

        for (zorder_id, cluster_id, block_id, ccs) in blocks:
            storage_id = uuid.uuid4()
            log.debug(" split : Create block {} {} -> {}".format(cluster_id, block_id, storage_id))
            token_split = []
            hash_key = cluster_id_to_token[cluster_id]
            log.debug(" storage_id {} -> hash_key {}".format(storage_id, hash_key))
            for t in self._tokens:
                if hash_key >= t[0] and hash_key < t[1]:
                    token_split.append(t)
                    break; # Finish for

            # build a view of self for the block NOT TRIVIAL :(
            pyccs = [ i * self._row_elem for i in ccs ]

            # Calculate the indexing part using slices array
            slc = [ slice(i, i + self._row_elem) for i in pyccs ]

            new_sn = self.view(np.ndarray)[tuple(slc)]  # The final view

            data_distrib = 0 # ZORDER_ALGORITHM
            if config.arrow_enabled:
                data_distrib = 3 # FORTRANORDER
            metas = HArrayMetadata(
                                   list(new_sn.shape),
                                   list(new_sn.strides),
                                   pyccs,
                                   new_sn.dtype.kind,
                                   new_sn.dtype.byteorder,
                                   new_sn.itemsize,
                                   new_sn.flags.num,
                                   data_distrib)

            # Store new metadata for the block into istorage
            new_args = self._build_args._replace(metas=metas, tokens=token_split, storage_id=storage_id)
            StorageNumpy._store_meta(new_args)

            #TODO: This reserve_numpy_array creates an HCACHE with the same name as the parent, therefore N HCACHES!!
            result = StorageNumpy.reserve_numpy_array(storage_id, self._name, metas) # storage_id is NOT used at all
            input_array = result[0]
            resultado = np.asarray(input_array).view(StorageNumpy)

            # Forge an IStorage(the order of these 3 instrucions is important)! Yolanda's eyes bleed!
            IStorage.__init__(resultado, tokens=token_split) #Without storageid to avoid Cassandra access
            resultado.storage_id = storage_id # Now set the storage_id
            IStorage.make_persistent(resultado, self._name) # And mark it as persistent

            resultado._numpy_full_loaded = False
            resultado._hcache = result[1]

            resultado._offsets = metas.offsets
            resultado._build_args = new_args

            resultado._row_elem = resultado._hcache.get_elements_per_row(storage_id, metas)
            resultado._all_coords = self._all_coords
            resultado._n_blocks = self._n_blocks

            # TODO: improve this implementation: goal destroy the twin, and reuse the parent twin
            resultado._twin_ref  = self._twin_ref
            resultado._twin_id   = self._twin_id
            resultado._twin_name = self._twin_name

            # TODO: Add a new column into hecuba.istorage with all the splits for the current storageid
            yield resultado

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
        Args:
            ksp : keyspace_arrow
            name: table_arrow
        Returns: keyspace_arrow.table_buffer
        """
        return ksp + "." + name[:-6] +"_buffer"

    @staticmethod
    def _comes_from_split(metas):
        # FIXME Use an 'offsets' value of None instead of this
        return (metas.offsets[0] != -1);    # Optimization: Just check the first coordinate

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
        log.debug("INITIALIZE EXISTING OBJECT name=%s sid=%s", name, storage_id)
        if not storage_id:  # StorageNumpy(None, name="xxxx", NONE)
            storage_id = storage_id_from_name(name)

        # Load metadata
        istorage_metas = get_istorage_attrs(storage_id)
        name = istorage_metas[0].name
        twin_id = istorage_metas[0].twin_id # This MUST match 'storage_id_from_name(_twin_name)'
        my_metas = istorage_metas[0].numpy_meta
        base_metas = my_metas
        base_numpy = istorage_metas[0].base_numpy


        if base_numpy is not None:
            # it is a view or a twin, therefore load the base instead of storage_id
            log.debug("Shared view of {}".format(base_numpy))
            base_metas = get_istorage_attrs(base_numpy)[0].numpy_meta

        tokens = istorage_metas[0].tokens

        # Reserve array
        # metas_to_reserve: If we come from a split we need to reserve space
        # just for one piece of the numpy: use the metadatas of the piece
        # otherwise we reserve space for the whole numpy (if we are a view we
        # use the metas of the base numpy)
        # metas_to_calculate: calculate_coords computes the indexes of a numpy.
        # This is used at getitem to compute the blocks to load (select_blocks)
        # If we come from a split the accesses to disk are relative to the
        # original object. For this reason we need to compute all_coords for
        # the original objet

        if StorageNumpy._comes_from_split(my_metas):
            metas_to_reserve   =  my_metas  # reserve only for the current piece
            metas_to_calculate =  base_metas # but keep the global metadata in memory
        else:
            metas_to_reserve   =  base_metas
            metas_to_calculate =  my_metas
        result = cls.reserve_numpy_array(storage_id, name, metas_to_reserve) # storage_id is NOT used at all
        input_array = result[0]
        obj = np.asarray(input_array).view(cls)
        IStorage.__init__(obj, name=name, storage_id=storage_id, tokens=tokens)
        obj._numpy_full_loaded = False
        obj._hcache = result[1]


        obj._offsets = metas_to_reserve.offsets
        obj._build_args = obj.args(obj.storage_id, istorage_metas[0].class_name,
                istorage_metas[0].name, metas_to_reserve, istorage_metas[0].block_id, base_numpy, twin_id,
                istorage_metas[0].tokens)
        if not StorageNumpy._comes_from_split(my_metas):    #If we come from a split, the twin will be shared with the parent!!!
            if config.arrow_enabled and twin_id is not None and my_metas.partition_type != 2:
                # Load TWIN array
                #print ("JJ __new__ twin_name ", obj._twin_name, flush=True);
                #print ("JJ __new__ twin_id ", obj._twin_id, flush=True);
                obj._twin_id   = twin_id
                obj._twin_name = StorageNumpy.get_arrow_name(obj._ksp, obj._table)
                twin = StorageNumpy._initialize_existing_object(cls, obj._twin_name, obj._twin_id)
                obj._twin_ref = twin
                twin._twin_id = obj.storage_id # Use the parent ID
        obj._row_elem = obj._hcache.get_elements_per_row(storage_id, metas_to_calculate)
        obj._calculate_coords(metas_to_calculate)
        #print (" JJ _initialize_existing_object name={} sid={} DONE".format(name, storage_id), flush=True)
        return obj


    def __new__(cls, input_array=None, name=None, storage_id=None, block_id=None, **kwargs):
        log.debug("input_array=%s name=%s storage_id=%s ",input_array is not None, name, storage_id)
        if name is not None:
            # Construct full qualified name to deal with cases where the name does NOT contain keyspace
            (ksp, table) = extract_ks_tab(name)
            name = ksp + "." + table
            if (len(table)>40 or table.startswith("HECUBA")):
                # Cassandra limits name to 48 characters: we reserve 8 characters
                # for special Hecuba tables
                raise AttributeError("The name of an user StorageNumpy is limited to 40 chars and can not start 'HECUBA' {}".format(table))

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
                obj = input_array.copy()
            else:
                # StorageNumpy(numpy, None, None)
                obj = np.asarray(input_array).copy().view(cls)
                if config.arrow_enabled and getattr(input_array, 'ndim', 0) == 2:
                    obj._twin_id  = None
                    obj._twin_ref = np.asarray(input_array).T.copy().view(cls)
                    log.debug("Created TWIN")
            IStorage.__init__(obj, name=name, storage_id=storage_id, kwargs=kwargs)

            if name or storage_id: # The object needs to be persisted
                load_data= (input_array is None) and (config.load_on_demand == False)
                if input_array is not None:
                    if isinstance(input_array,StorageNumpy):
                        log.warn("Creating a Persistent StorageNumpy.")
                    obj._persist_data(obj._get_name())
                if load_data: #FIXME aixo hauria d'afectar a l'objecte existent (aqui ja existeix a memoria... o hauria)
                    obj[:]	# HACK! Load ALL elements in memory NOW (recursively calls getitem)

        #print("JJ name = ", name, flush=True)
        #print("JJ _twin_name = ", obj._twin_name, flush=True)
        #print("JJ _name = ", obj._name, flush=True)
        return obj

    def __init__(self, input_array=None, name=None, storage_id=None, **kwargs):
        pass # DO NOT REMOVE THIS FUNCTION!!! Yolanda's eyes bleed!


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
            self._loaded_coordinates = getattr(obj, '_loaded_coordinates', [])
            if type(obj) == StorageNumpy:
                if obj.shape == self.shape:
                    self._numpy_full_loaded = obj._numpy_full_loaded
                else:
                    self._numpy_full_loaded = True # We come from a getitem and it has been already loaded
            else:
                # three cases: 1) StorageNumpy from a numpy (True) 2) split (False) or 3) instantiate an existing obj (False)
                # Is there any other?
                self._numpy_full_loaded = True # Default value

            self._is_persistent = getattr(obj, '_is_persistent', None)
            self._block_id = getattr(obj, '_block_id', None)
            self._twin_id   = getattr(obj, '_twin_id', None)
            self._twin_ref  = getattr(obj, '_twin_ref', None)
            self._twin_name = getattr(obj, '_twin_name', None)
            self._all_coords = getattr(obj, '_all_coords', None)
            self._n_blocks = getattr(obj, '_n_blocks', None)
            self._class_name = getattr(obj,'_class_name', 'hecuba.hnumpy.StorageNumpy')
            self._tokens = getattr(obj,'_tokens',None)
            self._offsets = getattr(obj,'_offsets', [-1] * self.ndim) # Initialize offsets to '-1'
            self._build_args = getattr(obj, '_build_args', None)
            #if objargs is None and self._is_persistent:
            #    hfetch_metas = HArrayMetadata(list(self.shape), list(self.strides),
            #                              self._offsets, self.dtype.kind, self.dtype.byteorder,
            #                              self.itemsize, self.flags.num, 0)
            #    objargs = self.args(self.storage_id, self._class_name, self._get_name(), hfetch_metas,
            #                            self._block_id, None, self._twin_id, self._tokens)
            #self._build_args = objargs
            #log.debug("  __array_finalize__ build_args.metas.offsets {}".format(self._build_args.metas.offsets))
            log.debug("  __array_finalize__ _offsets {}".format(self._offsets))
        else:
            log.debug("  __array_finalize__ copy")
            # Initialize fields as the __new__ case with input_array and not name
            self._loaded_coordinates = []
            self._numpy_full_loaded  = True # FIXME we only support copy for already loaded objects
            self._name               = None
            self.storage_id          = None
            self._is_persistent      = False
            self._twin_ref           = getattr(obj, '_twin_ref', None) # FIXME If it is a copy, it should copy the twin also(already done at copy... but move it here)
            self._offsets            = [-1] * self.ndim # Initialize offsets to '-1'
            self._class_name         = getattr(obj,'_class_name', 'hecuba.hnumpy.StorageNumpy')
            self._block_id           = getattr(obj, '_block_id', None)


    @staticmethod
    def _get_base_array(self):
        ''' Returns the 'base' numpy from this SN.  '''
        base = getattr(self, 'base',None)
        if base is None:
            base = self
        return base

    @staticmethod
    def _create_tables(name):
        (ksp, table) = extract_ks_tab(name)
        log.debug("Create table %s %s", ksp, table)
        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (ksp, config.replication)
        config.executelocked(query_keyspace)


        query_table='CREATE TABLE IF NOT EXISTS ' + ksp + '.' + table + '(storage_id uuid , '\
                                                                'cluster_id int, '           \
                                                                'block_id int, '             \
                                                                'payload blob, '             \
                                                                'PRIMARY KEY((storage_id,cluster_id),block_id))'
        config.executelocked(query_table)


    @staticmethod
    def _create_tables_arrow(name):
        (ksp, table) = extract_ks_tab(name)
        if config.arrow_enabled:
            # Add 'arrow' tables
            #	harrow_ to read
            #	buffer_ to write
            query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (ksp, config.replication)
            config.executelocked(query_keyspace)

            tbl_buffer = StorageNumpy.get_buffer_name(ksp, table)

            query_table_buff ='CREATE TABLE IF NOT EXISTS ' + tbl_buffer + \
                                                                    '(storage_id uuid , '    \
                                                                    'cluster_id  int, '      \
                                                                    'col_id      bigint, '   \
                                                                    'row_id      bigint, '   \
                                                                    'size_elem   int, '      \
                                                                    'payload     blob, '     \
                                                                    'PRIMARY KEY((storage_id, cluster_id), col_id))'
            config.executelocked(query_table_buff)
            query_table_arrow='CREATE TABLE IF NOT EXISTS ' + name + \
                                                                    '(storage_id uuid, '    \
                                                                    'cluster_id  int, '     \
                                                                    'col_id      bigint, '  \
                                                                    'arrow_addr  bigint, '  \
                                                                    'arrow_size  int, '     \
                                                                    'PRIMARY KEY((storage_id, cluster_id), col_id))'
            log.debug("Create table %s and %s", name, tbl_buffer)
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
                                    storage_args.twin_id,
                                    storage_args.tokens])

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


    @staticmethod
    def _add_offset(coord, offset):
        if coord < 0:
            new_coord = abs(coord)-1
        else:
            new_coord = coord
        return new_coord + offset

    def _adapt_coords(self, sliced_coord):
        '''
        Modify coordinates from __getitem__ parameter 'sliced_coord' to match the coordinates from a bigger base array.
        This is used when spliting bigger numpy, and a SN has been created for each block. The coordinates of these blocks
        start at 0, but they are relative to the bigger array. This function adapts the initial coordinate of the block to
        the sliced_coord.
        Args:
            sliced_coord:   The coordinates from the programmer (corresponding to current array) to adapt
        Returns adapted coordinates.

            istorage => (off1,off2,off3)
            *x[a]    scalar a + off1
            *x[:]    slice  a.start + off1, a.stop+off1
            *x[a,b,...]  tuple of ints  a + off1, b + off2
            *x[-a,-b,...]  tuple of ints  (abs(a)-1) + off1, (abs(b)-1) + off2 (ONLY INTERESTED IN FINAL BLOCK)
            *x[:,:,...]  tuple of slices a.start + off1, a.stop+off1, b.start+off2, b.stop+off2
            x[[..],[...]]   list  NOT SUPPORTED!
            x[0,:]

        '''
        if not StorageNumpy._comes_from_split(self._build_args.metas):
            return sliced_coord
        new_coord = sliced_coord
        if isinstance(sliced_coord, int):
            off1 = self._build_args.metas.offsets[0]
            new_coord = StorageNumpy._add_offset(sliced_coord, off1)

        elif isinstance(sliced_coord, slice):
            off1 = self._build_args.metas.offsets[0]
            new_coord = slice(StorageNumpy._add_offset(sliced_coord.start, off1), StorageNumpy._add_offset(sliced_coord.stop, off1))

        elif isinstance(sliced_coord, tuple):
            """ tuple of what? scalar, instances... ONLY SCALAR """
            n = len(sliced_coord)

            new_coord = []
            for i in range(n): # For each elemnt in tuple, increase by the corresponding offset
                off = self._build_args.metas.offsets[i]
                if isinstance(sliced_coord[i], int):
                    value = StorageNumpy._add_offset(sliced_coord[i], off)
                elif isinstance(sliced_coord[i], slice):
                    if sliced_coord[i] == slice(None, None, None):
                        value = slice(off, off+self._row_elem) ## WARNING this works due to the blocks being square (#rows == #cols)
                    else:
                        value = slice(StorageNumpy._add_offset(sliced_coord[i].start, off), StorageNumpy._add_offset(sliced_coord[i].stop, off))
                else:
                    raise NotImplementedError("Not supported type in tuple " + type(sliced_coord[i]))

                new_coord.append( value )

            new_coord = tuple(new_coord)

        else:
            raise NotImplementedError("NOT SUPPORTED type in storageNumpy idx"+type(sliced_coord))

        log.debug("adapted blocks :{} + {} -> {}".format(sliced_coord, self._build_args.metas.offsets, new_coord))
        return new_coord

    def _select_blocks(self,sliced_coord):
        """
            Calculate the list of block coordinates to load given a specific numpy slice syntax.
            Args:
                self: StorageNumpy to apply list of coordinates.
                sliced_coord: Slice syntax to evaluate.
            May raise an IndexError exception
        """
        if isinstance(sliced_coord, slice) and sliced_coord == slice(None, None, None):
            new_coords = None
            if StorageNumpy._comes_from_split(self._build_args.metas):
                offsets = [i // self._row_elem for i in self._offsets]
                new_coords = [tuple(offsets)] #Ask for the initial coordinate to load the block
        else:
            sliced_coord = self._adapt_coords(sliced_coord)
            new_coords = self._all_coords[sliced_coord] // self._row_elem
            new_coords = new_coords.reshape(-1, self.ndim)
            new_coords = frozenset(tuple(coord) for coord in new_coords)
            new_coords = list(new_coords)

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
            raise NotImplementedError("NOT SUPPORTED to load coordinates on a volatile object")

        load = True # By default, load everything
        if not new_coords: # Special case: Load everything
            log.debug("LOADING ALL BLOCKS OF NUMPY")
            self._numpy_full_loaded = True
            new_coords = None
            self._loaded_coordinates = None
        else:
            log.debug("LOADING COORDINATES")

            # coordinates is the union between the loaded coordinates and the new ones
            coordinates = list(set(it.chain.from_iterable((self._loaded_coordinates, new_coords))))
            if (len(coordinates) != len(self._loaded_coordinates)):
                self._numpy_full_loaded = (len(coordinates) == self._n_blocks)
                self._loaded_coordinates = coordinates
            else:
                load = False

        if load:
            base_numpy = StorageNumpy._get_base_array(self)
            from_split = StorageNumpy._comes_from_split(self._build_args.metas)
            if from_split:
                log.debug(" load_block from {} ".format(self._build_args.base_numpy))
                istorage_metas = get_istorage_attrs(self._build_args.base_numpy)
                metas = istorage_metas[0].numpy_meta
                self._numpy_full_loaded = True  # Mark the piece of numpy as loaded
                self._loaded_coordinates = None
            else:
                metas = self._build_args.metas
            self._hcache.load_numpy_slices([self._build_args.base_numpy], metas, [base_numpy],
                                   new_coords, from_split)

    def _select_columns(self, sliced_coord):
        """
        Returns None or a list of columns accesed
        """
        if not config.arrow_enabled:
            log.debug("HECUBA_ARROW is not enabled. Columnar acces disabled.")
            return None
        if self.ndim > 2:   # Not supported case. Only 2 dimensions!
            return None
        columns = None
        if isinstance(sliced_coord, slice) and sliced_coord == slice(None, None, None):
            if not StorageNumpy._comes_from_split(self._build_args.metas):
                log.debug("Access ALL matrix elements detected. Columnar access used")
                columns = list(range(self.shape[self.ndim-1]))
            else:
                columns =[]
                for c in range(self.shape[self.ndim-1]):
                    # Add the column offset...
                    columns.append(StorageNumpy._add_offset(c, self._build_args.metas.offsets[1]))

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
                    columns.append(self._adapt_coords(sliced_coord[-1])) #WARNING: Using adapt_coords... but it will work because the block is square (and number of rows and number of columns are equal)
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
            self._twin_ref._hcache.load_numpy_slices([self._build_args.base_numpy],
                                        self._twin_ref._build_args.metas,
                                        [base_numpy],
                                        columns,
                                        False)

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

            if type(values)==StorageNumpy and not values._numpy_full_loaded:
                values[:]  # LOAD the values as the numpy.__setitem__ will only use memory

            super(StorageNumpy, self).__setitem__(sliced_coord, values)

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
            return
        #if self._twin_ref is not None:
        #    super(StorageNumpy, self._twin_ref).__setitem__(sliced_coord, values)
        super(StorageNumpy, self).__setitem__(sliced_coord, values)
        return

    def _persist_data(self, name, formato=0):
        """
        Persist data to cassandra, the common attributes have been generated by IStorage.make_persistent
        Args:
            StorageNumpy to persist
            name to use
            [formato] to store the data (0-ZOrder, 2-columnar, 3-FortranOrder) # 0 ==Z_ORDER (find it at SpaceFillingCurve.h)
        """
        twin = self._twin_ref
        if twin is not None: # Persist Twin before current object (to obtain _twin_id)
            self._twin_name = StorageNumpy.get_arrow_name(self._ksp, self._table)
            IStorage.__init__(twin, storage_id=None, name=self._twin_name)
            self._twin_id   = twin.storage_id
            twin._twin_id = self.storage_id # Parent's ID
            twin._persist_data(self._twin_name, 2)

        if not getattr(self,'_built_remotely', None):
            if formato == 2:    # COLUMNAR
                self._create_tables_arrow(name)
            else :
                self._create_tables(name)

        if not getattr(self, '_hcache', None):
            self._hcache = self._create_hcache(name)

        if None in self or not self.ndim:
            raise NotImplemented("Empty array persistance")


        # Persist current object
        if formato == 0 and config.arrow_enabled: # If arrow & ZORDER -> FortranOrder
            formato = 3
        hfetch_metas = HArrayMetadata(list(self.shape), list(self.strides),
                                      list(self._offsets), self.dtype.kind, self.dtype.byteorder,
                                      self.itemsize, self.flags.num, formato)
        self._build_args = self.args(self.storage_id, self._class_name, self._get_name(), hfetch_metas, self._block_id,
                                     self.storage_id, # base_numpy is storage_id because until now we only reach this point if we are not inheriting from a StorageNumpy. We should update this if we allow StorageNumpy from volatile StorageNumpy
                                     getattr(self,'_twin_id', None),
                                     self._tokens)
        if len(self.shape) != 0:
            if formato == 2:    # If we are in columnar format we are the twin and _twin_id field contains the original storage_id of the parent
                sid = self._twin_id
            else:
                sid = self._build_args.base_numpy

            self._hcache.store_numpy_slices([sid], self._build_args.metas, [StorageNumpy._get_base_array(self)],
                                            None)
        StorageNumpy._store_meta(self._build_args)
        if formato != 2:
            self._row_elem = self._hcache.get_elements_per_row(self.storage_id, self._build_args.metas)
            self._calculate_coords(self._build_args.metas)


    def make_persistent(self, name):
        log.debug("Make %s persistent", name)

        super().make_persistent(name)
        self._persist_data(name)


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
                                           None, False)
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
        #FIXME if self is not full loaded... load it
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
