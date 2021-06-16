import itertools
import uuid
import pickle
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
                                                  '(storage_id, class_name, name, numpy_meta, block_id, base_numpy, twin_id, view_serialization, tokens)'
                                                  'VALUES (?,?,?,?,?,?,?,?,?)')

    args_names = ["storage_id", "class_name", "name", "metas", "block_id", "base_numpy", "twin_id", "view_serialization", "tokens"]
    args = namedtuple('StorageNumpyArgs', args_names)


    def getID(self):
        """
        Method to retrieve the storage id as string. Used by PyCOMPSs solely.
        if the StorageNumpy is a persistent slice create a this point the entry in the IStorage to avoid
        serialization and enhance locality
        :return: Storage_id as str
        """
        sid = super().getID()
        if sid != 'None' and self._persistance_needed:
            # build args should contain the right metas: calculate them at getitem
            # we should avoid the store meta for the SN that has been persisted through a make_persistent.
            # We mark it as persistentance_needed=true at getitem and at this point create the entry
            StorageNumpy._store_meta(self._build_args)
            self._persistance_needed = False
        return sid


    def _calculate_nblocks(self, metas):
        ''' Calculate the number of blocks used by 'metas' (aka the number of blocks reserved in memory)
            This is used to determine if a numpy is full loaded on memory and avoid accesses to cassandra.
            Args:
                self : object to add new coords
                metas: metadatas to use to calculate the blocks
        '''

        log.debug("JCOSTA _calculate_nblocks ENTER sid={} row_elem={}".format(self.storage_id, self._row_elem))
        ndim = len(metas.dims)
        num_blocks = 1
        for i in range(0, ndim):
            b = ceil(metas.dims[i] / self._row_elem)
            num_blocks = num_blocks * b
        self._n_blocks = num_blocks
        log.debug("JCOSTA _calculate_nblocks sid={} _n_blocks={}".format(self.storage_id, self._n_blocks))

    # used by dislib? To be deleted after checking this
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
        bytes_cluster_id = bytes([0, 4]) + cluster_id.to_bytes(4,'big') + bytes([0])
        mykey = bytes_storage_id + bytes_cluster_id
        return murmur3.murmur3(mykey)


    def split(self,cols=True):
        """
        Divide numpy into persistent views to exploit parallelism.

        cols: Use a division by columns to exploit arrow. If False, the numpy is divided into the inner blocks stored in cassandra.
        """
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

        splitted_block={}   # list of blocks per (cluster_id,block_id)
        tokens={}   # list of tokens per cluster_id
        for (zorder_id, cluster_id, block_id, ccs) in blocks:
            log.debug(" split : Create block {} {} ".format(cluster_id, block_id ))
            token_split = []
            hash_key = cluster_id_to_token[cluster_id]
            for t in self._tokens:
                if hash_key >= t[0] and hash_key < t[1]:
                    token_split.append(t)
                    break; # Finish for
            if cluster_id not in tokens:
                tokens[cluster_id] = token_split

            ################
#
#            +-----------------+
#            | 0,0 | 0,1 | 0,2 |
#            +-----------------+
#            | 1,0 | 1,1 | 1,2 |
#            +-----------------+
#            ...
#            +-----------------+

            if cols:
                mykey=(cluster_id,0)    # Group the blockIDs inside the same clusterID
            else:
                mykey=(cluster_id,block_id)
            if mykey not in splitted_block:
                splitted_block[mykey] = []
            splitted_block[mykey].append(ccs)
            #splitted_block[0] = [ (0,0), (1,0), ... (N, 0) ] --> (slice(none,none,none), 0)


        for mykey, values in splitted_block.items():
#            +----------------------+
#            | 0,0  | 0,22  | 0,44  |
#            +----------------------+
#            | 22,0 | 22,22 | 22,44 |
#            +----------------------+
#            ...
#            +----------------------+

#            +======----------------+
#            I 0,0  I 0,22  | 0,44  |
#            +----------------------+
#            I 22,0 I 22,22 | 22,44 |
#            +----------------------+
#            ...
#            +----------------------+
#            I 22,0 I 22,22 | 22,44 |
#            +======----------------+

            cluster_id = mykey[0]
            # Calculate the indexing part using slices array
            if not cols:
                ccs = values[0]
                # 'values' contains block_coords that must be transformed to original_coordinates
                pyccs = [ i * self._row_elem for i in ccs]
                slc = [ slice(i, i + self._row_elem) for i in pyccs ]
                slc = tuple(slc)
            else:
                # Columns case: clusterID MUST match columnID
                for i in values:
                    if i[1] != cluster_id:
                        raise ValueError("OOOPS! ClusterID does not match with columnID")
                slc = ( slice(None,None,None), slice(cluster_id*self._row_elem, cluster_id*self._row_elem + self._row_elem ) )

            token_split = tokens[cluster_id]

            self._last_sliced_coord = slc # HACK to call '_create_lazy_persistent_view' in 'array_finalize' when calling the next '__getitem__'
            resultado = super(StorageNumpy, self).__getitem__(slc) # Generate view in memory
            resultado._build_args = resultado._build_args._replace(tokens=token_split)

            # TODO: Add a new column into hecuba.istorage with all the splits for the current storageid
            yield resultado

#
#            data_distrib = 0 # ZORDER_ALGORITHM
#            if config.arrow_enabled:
#                data_distrib = 3 # FORTRANORDER
#            metas = HArrayMetadata(
#                                   list(new_sn.shape),
#                                   list(new_sn.strides),
#                                   new_sn.dtype.kind,
#                                   new_sn.dtype.byteorder,
#                                   new_sn.itemsize,
#                                   new_sn.flags.num,
#                                   data_distrib)

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
        metas_to_reserve = my_metas
        base_numpy = istorage_metas[0].base_numpy


        if storage_id != base_numpy:
            # it is a view or a twin, therefore load the base instead of storage_id
            # base_numpy can be None?
            log.debug("Shared view of {}".format(base_numpy))
            metas_to_reserve = get_istorage_attrs(base_numpy)[0].numpy_meta

        tokens = istorage_metas[0].tokens

        # Reserve array: even if we are a view we reserve space for the WHOLE numpy, as the memory

        result = cls.reserve_numpy_array(storage_id, name, metas_to_reserve) # storage_id is NOT used at all

        input_array = result[0]

        obj = np.asarray(input_array).view(cls) # This must be done BEFORE the view, to keep the BASE numpy loaded (otherwise the 'base' field is overwritten with the base of the view)

        # The data recovered from the istorage is a persistent view, therefore reconstruct the view
        myview = pickle.loads(istorage_metas[0].view_serialization)
        log.debug(" view of {}".format(myview))
        if isinstance(myview, tuple):
            obj = super(StorageNumpy, obj).__getitem__(myview)
        else:
            raise TypeError(" WARNING: recovered 'view_serialization' has unexpected type ", type(myview))


        IStorage.__init__(obj, name=name, storage_id=storage_id, tokens=tokens)

        obj._numpy_full_loaded = False
        obj._hcache = result[1]


        obj._build_args = obj.args(obj.storage_id, istorage_metas[0].class_name,
                istorage_metas[0].name, my_metas, istorage_metas[0].block_id, base_numpy, twin_id,
                myview,
                istorage_metas[0].tokens)
        if config.arrow_enabled and twin_id is not None and my_metas.partition_type != 2:
            # Load TWIN array
            obj._twin_id   = twin_id
            obj._twin_name = StorageNumpy.get_arrow_name(obj._ksp, obj._table)
            twin = StorageNumpy._initialize_existing_object(cls, obj._twin_name, obj._twin_id)
            obj._twin_ref = twin
            twin._twin_id = obj.storage_id # Use the parent ID
        obj._row_elem = obj._hcache.get_elements_per_row(storage_id, metas_to_reserve)
        obj._calculate_nblocks(my_metas)
        return obj


    def __new__(cls, input_array=None, name=None, storage_id=None, block_id=None, **kwargs):
        log.debug("input_array=%s name=%s storage_id=%s ENTER ",input_array is not None, name, storage_id)
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
        log.debug("input_array=%s name=%s storage_id=%s ENTER ",input_array is not None, name, storage_id)
        return obj

    def __init__(self, input_array=None, name=None, storage_id=None, **kwargs):
        pass # DO NOT REMOVE THIS FUNCTION!!! Yolanda's eyes bleed!

    @staticmethod
    def removenones(n, maxstop=None):
        """
            Remove the Nones from a slice:
            start -> 0
            stop  -> None or maxstop
            step  -> 1
            This is a helper utility for the later operations of addition
        """
        if not n.start:
            oldstart = 0
        else:
            oldstart = n.start
        if not n.step:
            oldstep = 1
        else:
            oldstep=n.step
        if not n.stop:
            newstop = maxstop
        else:
            newstop = n.stop
        return slice(oldstart, newstop, oldstep)

    def calculate_block_coords(self, view):
        """
        Return a list with all the block coordinates relative to 'self.base.shape' corresponding to the elements in 'view'
        """
        first=[]
        last=[]
        shape = self.base.shape
        SIZE= self._row_elem # 7 # ROW_ELEM
        for idx, i in enumerate(view):
            #print(" {}:  element ={} ".format(idx, i))
            if isinstance(i, int):
                first.append(i//SIZE)
                last.append(i//SIZE)
            else: # It's a slice
                n = StorageNumpy.removenones(i, shape[idx])
                print(" {}:    n ={} ".format(idx, n))
                first.append(n.start//SIZE)
                last.append((n.stop-1)//SIZE)
        #print(" first ={} last = {}".format(first,last))
        l=[]
        for i in range(len(view)):
            l.append( range(first[i], last[i]+1))

        return [b for b in itertools.product(*l)]

    @staticmethod
    def view_composer_internal(shape, old, new):
        """
            shape: tuple with the dimensions of the numpy with the 'old' view where the 'new' will be applied
            old  : cumulated view
            new  : MUST be a tuple with the new view to compose on top of old
        """
        if len(old) != len(shape):
            raise TypeError("old needs to be a tuple with the same dimensions ({}) as the numpy ({})".format(len(old), len(shape)))
            return

        if isinstance(new, tuple):
            # Recursive case
            old=list(old)
            j=0 # current old shape
            i=0 # current new element
            while i<len(new) and j < len(old):
                n = new[i]
                while isinstance(old[j],int) and j < len(old): #skip integers
                    j=j+1
                if j < len(old):
                    old[j] = StorageNumpy.view_composer_internal((shape[j],),(old[j],), n)
                    j=j+1
                i=i+1

            #print(" view_composer: ======> {}".format(old))
            return tuple(old)

        # Base cases
        #print(" view_composer: shape={} old={} new={}".format(shape, old, new))
        old0 = old[0]
        res = None
        if   isinstance(new, int):
            if isinstance(old0, int):
                #res = new + old0
                res = old0
            elif isinstance(old0, slice):
                old0 = StorageNumpy.removenones(old0, shape[0])
                if new < 0:
                    new = (old0.stop-old0.start) + new
                res = new*old0.step+old0.start
            else:
                raise NotImplementedError("Compose an int and a {}".format(type(old0)))

        elif isinstance(new, slice):
            new = StorageNumpy.removenones(new, shape[0])
            newstep = new.step
            if isinstance(old0, int):
                #res = old0*new.step + new.start
                res = old0
            elif isinstance(old0, slice):
                newstart=new.start
                newstop=new.stop
                old0 = StorageNumpy.removenones(old0, shape[0])
                if newstart<0:
                    #newstart=shapenew[0]+newstart
                    newstart = (old0.stop - old0.start) + newstart
                if newstop<0:
                    #newstop=shapenew[0]+newstop
                    newstop = (old0.stop - old0.start) + newstop
                oldstep = old0.step
                if oldstep > 1 and newstep > 1:
                    resstep = oldstep + newstep
                else:
                    resstep=max(oldstep,newstep)
                res = slice(old0.start+(newstart*oldstep), min(old0.start+(newstop*oldstep),old0.stop), resstep)
            else:
                raise NotImplementedError("Compose an slice and a {}".format(type(old0)))

        else:
            raise NotImplementedError("Compose an {} with previous slice".format(type(new)))

        if len(old) > 1:
            toreturn=list(old)
            toreturn[0]=res
            res =  tuple(toreturn)
        #print(" view_composer: ====> {}".format(res))
        return res


    def _view_composer_new(self, new_view):
        """
            Compose a view on top of self.base equivalent to 'new_view' on current object
        """
        print(" view_composer: shape={} old={} new={}".format(self.base.shape, self._build_args.view_serialization, new_view))
        if isinstance(new_view, int) or isinstance(new_view,slice):
            new_view=(new_view,)
        elif not isinstance (new_view,tuple):
            raise TypeError("View must be a tuple,int or slice instead of {}".format(type(new_view)))

        old = self._build_args.view_serialization
        #res = StorageNumpy.view_composer_internal(self.base.shape, old, self.shape, new_view)
        res = StorageNumpy.view_composer_internal(self.base.shape, old, new_view)
        print(" view_composer: ======> {}".format(res))
        return  res

    def _create_lazy_persistent_view(self, view):
        """
            Create a persistent view of current object.
            The resulting view, even it has an storage_id, is NOT persistent.
            It will be made persistent when 'getID()' is invoked on it (this
            will usually happen automatically when using COMPSS)
        """

        new_view_serialization = self._view_composer_new(view)

        storage_id = uuid.uuid4()
        self.storage_id = storage_id
        metas = HArrayMetadata(
                   list(self.shape),
                   list(self.strides),
                   self.dtype.kind,
                   self.dtype.byteorder,
                   self.itemsize,
                   self.flags.num,
                   self._build_args.metas.partition_type)

        new_args = self._build_args._replace(metas=metas, storage_id=storage_id,
                                             view_serialization=new_view_serialization)
        self._build_args = new_args
        self._calculate_nblocks(self._build_args.metas)
        self._persistance_needed = True

    # used as copy constructor
    def __array_finalize__(self, obj):
        if obj is None:
            log.debug("  __array_finalize__ NEW")
            return
        log.debug("__array_finalize__ self.base=None?%s obj.base=None?%s", getattr(self, 'base', None) is None, getattr(obj, 'base', None) is None)
        if self.base is not None: # It is a view, therefore, copy data from object
            log.debug("  __array_finalize__ view (new_from_template/view)")

            self.storage_id = getattr(obj, 'storage_id', None)
            print("JCOSTA __array_finalize__: {}".format(self.__dict__), flush=True)
            self._name = getattr(obj, '_name', None)
            self._hcache = getattr(obj, '_hcache', None)
            self._row_elem = getattr(obj, '_row_elem', None)
            # if we are a view we have ALREADY loaded all the subarray
            self._loaded_coordinates = getattr(obj, '_loaded_coordinates', [])
            self._is_persistent = getattr(obj, '_is_persistent', False)
            self._block_id = getattr(obj, '_block_id', None)
            self._twin_id   = getattr(obj, '_twin_id', None)
            self._twin_ref  = getattr(obj, '_twin_ref', None)
            self._twin_name = getattr(obj, '_twin_name', None)
            self._class_name = getattr(obj,'_class_name', 'hecuba.hnumpy.StorageNumpy')
            self._tokens = getattr(obj,'_tokens',None)
            self._build_args = getattr(obj, '_build_args', None)
            self._persistance_needed = getattr(obj, '_persistance_needed', False)

            if type(obj) == StorageNumpy: # Instantiate or getitem/split
                print("JCOSTA array_finalize obj == StorageNumpy", flush=True)
                if obj.shape == self.shape:
                    print("JCOSTA array_finalize obj.shape == self.shape", flush=True)
                    self._numpy_full_loaded = obj._numpy_full_loaded
                    self._n_blocks = getattr(obj, '_n_blocks', None)
                else:
                    print("JCOSTA array_finalize obj.shape != self.shape", flush=True)
                    if getattr(obj, '_last_sliced_coord', None):
                        # User is doing a getitem on a persistent object and
                        # the shape has changed, therefore a new entry in the
                        # istorage MAY BE  needed (in case she wants to load it
                        # remotely)
                        # NOTE: We do NOT allow persistent views for another persistent view
                        #       (where the base numpy differs from storageID)

                        self._create_lazy_persistent_view(obj._last_sliced_coord)
                        obj._last_sliced_coord = None


                    #self._numpy_full_loaded = True # We come from a getitem and it has been already loaded
                self._numpy_full_loaded = getattr(obj, '_numpy_full_loaded', False)
            else:
                # StorageNumpy from a numpy
                print("JCOSTA array_finalize obj != StorageNumpy", flush=True)
                self._numpy_full_loaded = True # Default value
        else:
            log.debug("  __array_finalize__ copy")
            # Initialize fields as the __new__ case with input_array and not name
            self._loaded_coordinates = []
            self._numpy_full_loaded  = True # FIXME we only support copy for already loaded objects
            self._name               = None
            self.storage_id          = None
            self._is_persistent      = False
            # If it is a copy, it should copy the twin also
            if getattr(obj, '_twin_ref', None) is not None:
                self._twin_id = None
                self._twin_name = None
                self._twin_ref = super(StorageNumpy, obj._twin_ref).copy()
            self._class_name         = getattr(obj,'_class_name', 'hecuba.hnumpy.StorageNumpy')
            self._block_id           = getattr(obj, '_block_id', None)
            self._persistance_needed = False


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
                                    pickle.dumps(storage_args.view_serialization),
                                    storage_args.tokens])

        except Exception as ex:
            log.warn("Error creating the StorageNumpy metadata with args: %s" % str(storage_args))
            raise ex

    @staticmethod
    def reserve_numpy_array(storage_id, name, metas):
        '''Provides a numpy array with the number of elements obtained through storage_id'''
        log.debug(" Reserve memory for {} {} {}".format(name, storage_id, metas))
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
                sliced_coord: Slice syntax to evaluate.
            May raise an IndexError exception
        """

        sliced_coord = self._view_composer_new(sliced_coord)
        new_coords = self.calculate_block_coords(sliced_coord)
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
            coordinates = list(set(itertools.chain.from_iterable((self._loaded_coordinates, new_coords))))
            if (len(coordinates) != len(self._loaded_coordinates)):
                self._numpy_full_loaded = (len(coordinates) == self._n_blocks)
                self._loaded_coordinates = coordinates
            else:
                load = False

        if load:
            base_numpy = StorageNumpy._get_base_array(self)
            if (self.storage_id != self._build_args.base_numpy):
                #We are a view and therefore we need the metas from the base numpy
                log.debug(" load_block from {} ".format(self._build_args.base_numpy))
                istorage_metas = get_istorage_attrs(self._build_args.base_numpy)
                metas = istorage_metas[0].numpy_meta
            else:
                metas = self._build_args.metas
            log.debug("  COORDINATES ARE {} ".format(new_coords))
            self._hcache.load_numpy_slices([self._build_args.base_numpy], metas, [base_numpy],
                                   new_coords)

    def is_columnar(self,sliced_coord):
        if not config.arrow_enabled:
            log.debug("HECUBA_ARROW is not enabled. Columnar acces disabled.")
            return False
        if self.ndim != 2:   # Not supported case. Only 2 dimensions!
            return False

        if isinstance(sliced_coord, slice) and sliced_coord == slice(None, None, None):
            return True
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
                        return False
                    return True
            return False
        return False

    def _select_columns(self, sliced_coord):
        """
        Returns None or a list of columns accesed
        """
        columns = None
        sliced_coord = self._view_composer_new(sliced_coord)
        last = sliced_coord[-1]
        if isinstance (last,int):
            columns = [last]
        else: # it is an slice
            last = StorageNumpy.removenones(last, self.shape[1])
            columns = [ c for c in range(last.start, last.stop, last.step)]

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
                                        columns)
            # Copy EACH element from one matrix to the other (taking into account that one matrix is the transpose from the other)
            # TODO figure a better way to do this
            for i in range(self._build_args.metas.dims[0]):
                for j in columns:
                    self.base.data[i,j] = self._twin_ref.data[j,i]
            if len(columns) == self._build_args.metas.dims[1]: #if we are loading ALL columns, we COPY the twin to the normal (2 copies)
                log.debug("LOADED WHOLE ARRAY ")
                self._numpy_full_loaded = True

    def _select_and_load_blocks(self, sliced_coord):
        if self._is_persistent:
            if not self._numpy_full_loaded:
                block_coord = self._select_blocks(sliced_coord)
                self._load_blocks(block_coord)

    def __getitem__(self, sliced_coord):
        log.info("RETRIEVING NUMPY {} is_persistent {}".format(sliced_coord, self._is_persistent))
        if self._is_persistent:

            self._last_sliced_coord = sliced_coord  # Remember the last getitem parameter, because it may force a new entry in the istorage at array_finalize

            #if the slice is a npndarray numpy creates a copy and we do the same
            if isinstance(sliced_coord, np.ndarray): # is there any other slicing case that needs a copy of the array????
                result = self.view(np.ndarray)[sliced_coord] # TODO: If self is NOT loaded LOAD IT ALL BEFORE
                return StorageNumpy(result) # Creates a copy (A StorageNumpy from a Numpy)

            if self._build_args.metas.partition_type == 2 :
                #HACK if the accessed numpy is a columnar one, assume that it is already in memory
                log.warn("Accessing a twin directly. Assuming it is already in memory")
                return super(StorageNumpy, self).__getitem__(sliced_coord)

        # Check if the access is columnar...
            if self.is_columnar(sliced_coord):
                columns = self._select_columns(sliced_coord)
                if columns is not None : # Columnar access
                    self._load_columns(columns)
            else: # Normal array access...
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
            if (self.storage_id != self._build_args.base_numpy):
                #We are a view and therefore we need the metas from the base numpy
                log.debug(" load_block from {} ".format(self._build_args.base_numpy))
                istorage_metas = get_istorage_attrs(self._build_args.base_numpy)
                metas = istorage_metas[0].numpy_meta
            else:
                metas = self._build_args.metas
            self._hcache.store_numpy_slices([self._build_args.base_numpy],
                    metas, [base_numpy],
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
        log.debug("_persist_data: {} format={} ENTER ".format(name, formato))
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
                                      self.dtype.kind, self.dtype.byteorder,
                                      self.itemsize, self.flags.num, formato)
        self._build_args = self.args(self.storage_id, self._class_name, self._get_name(), hfetch_metas, self._block_id,
                                     self.storage_id, # base_numpy is storage_id because until now we only reach this point if we are not inheriting from a StorageNumpy. We should update this if we allow StorageNumpy from volatile StorageNumpy
                                     getattr(self,'_twin_id', None),
                                     tuple([slice(None,None,None)]*self.base.ndim),  #We are a view of everything
                                     self._tokens)
        if len(self.shape) != 0:
            if formato == 2:    # If we are in columnar format we are the twin and _twin_id field contains the original storage_id of the parent
                sid = self._twin_id
            else:
                sid = self._build_args.base_numpy

            self._hcache.store_numpy_slices([sid], self._build_args.metas, [StorageNumpy._get_base_array(self)], # CHECK metas del padre i memoria tienen que coincidir
                                            None)
        StorageNumpy._store_meta(self._build_args)
        if formato != 2:
            self._row_elem = self._hcache.get_elements_per_row(self.storage_id, self._build_args.metas)
        log.debug("_persist_data: {} format={}".format(name, formato))


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
        metas = None
        if self._is_persistent and len(self.shape) and self._numpy_full_loaded is False:
            if (self.storage_id != self._build_args.base_numpy):
                #We are a view and therefore we need the metas from the base numpy
                istorage_metas = get_istorage_attrs(self._build_args.base_numpy)
                metas = istorage_metas[0].numpy_meta
            else:
                metas = self._build_args.metas
            log.debug(" UFUNC({}) load_block from {} ".format(method, metas))
            self._hcache.load_numpy_slices([self._build_args.base_numpy], metas, [base_numpy],
                                           None)
        results = super(StorageNumpy, self).__array_ufunc__(ufunc, method,
                                                            *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            return

        if self._is_persistent and len(self.shape):
            readonly_methods = ['mean', 'sum', 'reduce'] #methods that DO NOT modify the original memory, and there is NO NEED to store it
            if method not in readonly_methods:
                if metas is None: # Ooops... numpy is fully loaded and the metas calculation was skipped
                    if (self.storage_id != self._build_args.base_numpy):
                        #We are a view and therefore we need the metas from the base numpy
                        istorage_metas = get_istorage_attrs(self._build_args.base_numpy)
                        metas = istorage_metas[0].numpy_meta
                    else:
                        metas = self._build_args.metas
                    log.debug(" UFUNC({}) store_block from {} ".format(method, metas))

                self._hcache.store_numpy_slices([self._build_args.base_numpy], metas, [base_numpy],
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
        return n_sn


    ###### INTERCEPTED FUNCTIONS #####
    def dot(a, b, out=None):
        if isinstance(a, StorageNumpy) and a._is_persistent and not a._numpy_full_loaded:
           a[:]	# HACK! Load ALL elements in memory NOW (recursively calls getitem)

        if isinstance(b, StorageNumpy) and b._is_persistent and not b._numpy_full_loaded:
           b[:]	# HACK! Load ALL elements in memory NOW (recursively calls getitem)

        return config.intercepted['dot'](a,b,out) # At the end of this 'copy' is called
