import itertools
import uuid
import pickle
from collections import namedtuple
from typing import Tuple

from math import ceil

import numpy as np
from hecuba.hfetch import HNumpyStore, HArrayMetadata

from . import config, log, Parser
from .IStorage import IStorage
from .tools import extract_ks_tab, get_istorage_attrs, storage_id_from_name, build_remotely


class StorageNumpy(IStorage, np.ndarray):
    USE_FORTRAN_ACCESS=False
    BLOCK_MODE = 1
    COLUMN_MODE = 2
    _build_args = None

    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba.istorage'
                                                  '(storage_id, class_name, name, numpy_meta, block_id, base_numpy, view_serialization, tokens)'
                                                  'VALUES (?,?,?,?,?,?,?,?)')

    args_names = ["storage_id", "class_name", "name", "metas", "block_id", "base_numpy", "view_serialization", "tokens"]
    args = namedtuple('StorageNumpyArgs', args_names)


    def getID(self):
        """
        Method to retrieve the storage id as string. Used by PyCOMPSs solely.
        if the StorageNumpy is a persistent slice create a this point the entry in the IStorage to avoid
        serialization and enhance locality
        :return: Storage_id as str
        """
        sid = super().getID()
        if sid != 'None':
            if self._persistance_needed:
                # build args should contain the right metas: calculate them at getitem
                # we should avoid the store meta for the SN that has been persisted through a make_persistent.
                # We mark it as persistentance_needed=true at getitem and at this point create the entry
                StorageNumpy._store_meta(self._build_args)
                self._persistance_needed = False
            self.sync() # Data may be needed in another node, flush data
        return sid


    def _calculate_nblocks(self, view):
        ''' Calculate (and set) the number of used blocks in data storage by 'view'
            (aka the number of blocks reserved in memory)
            This is used to determine if a numpy is full loaded on memory and avoid accesses to cassandra.
            Args:
                self : object to use
                view : view used to calculate the blocks
        '''

        l = self.calculate_list_of_ranges_of_block_coords(view)
        num_blocks = 1
        for i in l:
            num_blocks = num_blocks * len(i)
        self._n_blocks = num_blocks
        log.debug("JCOSTA _calculate_nblocks sid={} _n_blocks={}".format(self.storage_id, self._n_blocks))

    # used by dislib? To be deleted after checking this
    def np_split(self, block_size: Tuple[int, int]):
        # For now, only split in two dimensions is supported
        bn, bm = block_size
        for block_id, i in enumerate(range(0, self.shape[0], bn)):
            block = [self[i: i + bn, j:j + bm] for j in range(0, self.shape[1], bm)]
            obj = self.__class__(input_array=block, name=self._get_name(), storage_id=uuid.uuid4(), block_id=block_id)
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


    def split(self,cols=None):
        """
        Divide numpy into persistent views to exploit parallelism.

        cols: Decides how to divide the numpy:
                If None, use the inner blocks stored in cassandra.
                If True, divide by columns of blocks (this allows to exploit arrow when enabled)
                If False, divide by rows of blocks.
        """
        # TODO this should work for VOLATILE objects too! Now only works for PERSISTENT
        if self._build_args.metas.partition_type == 2:
            raise NotImplementedError("Split on columnar data is not supported")


        tokens = self._get_tokens(cols)
        log.debug("split: shape %s cols %s", self.shape, cols)
        if cols is True:
            return self._split_by_cols(tokens)
        if cols is False:
            return self._split_by_rows(tokens)
        return self._split_by_blocks(tokens)

    def _get_tokens(self, cols):

        exploit_locality = False
        if StorageNumpy._arrow_enabled(self): #Fortran order
            exploit_locality = (cols != False) # By blocks or by columns
        else: #Zorder
            exploit_locality = (cols == None) # By blocks only

        tokens = None
        if exploit_locality:
            # Calculate the blocks of the numpy
            blocks = self._hcache.get_block_ids(self._build_args.metas) # returns a list of tuples (cluster_id, block_id)

            #Build map cluster_id -> token
            cluster_id_to_token = {}
            for (zorder_id, cluster_id, block_id, ccs) in blocks:
                if not cluster_id_to_token.__contains__(cluster_id):
                    hash_key = StorageNumpy._composite_key(self.storage_id, cluster_id)
                    cluster_id_to_token[cluster_id] = hash_key

            #Calculate the tokens for each block
            tokens = {}
            for (zorder_id, cluster_id, block_id, ccs) in blocks:
                log.debug(" split : Create block {} {} ".format(cluster_id, block_id ))
                if cluster_id not in tokens:
                    token_split = []
                    hash_key = cluster_id_to_token[cluster_id]
                    for t in self._tokens:
                        if hash_key >= t[0] and hash_key < t[1]:
                            token_split.append(t)
                            break; # Finish for
                    tokens[cluster_id] = token_split

        return tokens

    def _split_by_blocks(self, tokens):

        blocks = self._hcache.get_block_ids(self._build_args.metas) # returns a list of tuples (cluster_id, block_id)
        _parent_numpy_full_loaded=self._numpy_full_loaded
        for (zorder_id, cluster_id, block_id, ccs) in blocks:
            # 'values' contains block_coords that must be transformed to original_coordinates
            pyccs = [ i * self._row_elem for i in ccs]
            slc = [ slice(i, i + self._row_elem) for i in pyccs ]
            slc = tuple(slc)
            token_split = tokens[cluster_id]

            self._last_sliced_coord = slc # HACK to call '_create_lazy_persistent_view' in 'array_finalize' when calling the next '__getitem__'
            resultado = super(StorageNumpy, self).__getitem__(slc) # Generate view in memory
            resultado._numpy_full_loaded = _parent_numpy_full_loaded # Due to the HACK, we need to keep the _numpy_full_loaded status
            resultado._build_args = resultado._build_args._replace(tokens=token_split)

            yield resultado
#            ################
##            ccs: Index of blocks
##            +-----------------+
##            | 0,0 | 0,1 | 0,2 |
##            +-----------------+
##            | 1,0 | 1,1 | 1,2 |
##            +-----------------+
##            ...
##            +-----------------+
#
#             pyccs: initial coordinates of each block
##            +======----------------+
##            I 0,0  I 0,22  | 0,44  |
##            +----------------------+
##            I 22,0 I 22,22 | 22,44 |
##            +----------------------+
##            ...
##            +----------------------+
##            I 22,0 I 22,22 | 22,44 |
##            +======----------------+


    def _split_by_cols(self, mytokens):
        """
        Generator to divide numpy in blocks of columns (taking into account how the data is stored in disk)
        """
        log.debug(" split_by_cols shape:%s row_elem:%s ", self.shape, self._row_elem)
        list_of_clusters= range(0, self.shape[1], self._row_elem)

        _parent_numpy_full_loaded=self._numpy_full_loaded
        for cluster_id in list_of_clusters:
            log.debug(" split_by_cols cluster_id: %s", cluster_id)
            slc = ( slice(None,None,None), slice(cluster_id, cluster_id + self._row_elem ) )


            self._last_sliced_coord = slc # HACK to call '_create_lazy_persistent_view' in 'array_finalize' when calling the next '__getitem__' (we want to AVOID calling 'getitem' directly because it LOADS data)
            resultado = super(StorageNumpy, self).__getitem__(slc) # Generate view in memory
            resultado._numpy_full_loaded = _parent_numpy_full_loaded # Due to the HACK, we need to keep the _numpy_full_loaded status
            if mytokens is not None:
                resultado._build_args = resultado._build_args._replace(tokens=mytokens[cluster_id//self._row_elem])

            yield resultado

    def _split_by_rows(self, tokens):
        """
        Generator to divide numpy in blocks of columns (taking into account how the data is stored in disk)
        """
        log.debug(" split_by_cols shape:%s row_elem:%s ", self.shape, self._row_elem)
        list_of_clusters= range(0, self.shape[0], self._row_elem)

        _parent_numpy_full_loaded=self._numpy_full_loaded
        for cluster_id in list_of_clusters:
            log.debug(" split_by_cols cluster_id: %s", cluster_id)
            slc = ( slice(cluster_id, cluster_id + self._row_elem ), slice(None,None,None) )

            self._last_sliced_coord = slc # HACK to call '_create_lazy_persistent_view' in 'array_finalize' when calling the next '__getitem__' (we want to AVOID calling 'getitem' directly because it LOADS data)
            resultado = super(StorageNumpy, self).__getitem__(slc) # Generate view in memory
            resultado._numpy_full_loaded = _parent_numpy_full_loaded # Due to the HACK, we need to keep the _numpy_full_loaded status
            # TOKENS are ignored in this case

            yield resultado

    @staticmethod
    def get_arrow_name(name):
        # get_arrow_name: Returns the keyspace and table name of the arrow table (READ) of table name
        (ksp,table) = extract_ks_tab(name)
        return ksp + "_arrow." + table[:42] +"_arrow"

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
        if name and storage_id:
            log.warning("INITIALIZE EXISTING OBJECT request passing both name {} and storage_id {}. Ignoring parameter 'name'".format(name, storage_id))
        if not storage_id:  # StorageNumpy(None, name="xxxx", NONE)
            storage_id = storage_id_from_name(name)

        # Load metadata
        istorage_metas = get_istorage_attrs(storage_id)
        if len(istorage_metas) == 0:
            msg = "Persistent StorageNumpy Storage_id={}".format(storage_id)
            if name:
                msg = msg + " name={}".format(name)
            raise ValueError("{} does not exist".format(msg))
        name = istorage_metas[0].name
        my_metas = istorage_metas[0].numpy_meta
        metas_to_reserve = my_metas
        base_numpy = istorage_metas[0].base_numpy


        if storage_id != base_numpy:
            # it is a view load the base instead of storage_id
            # base_numpy can be None?
            log.debug("Shared view of {}".format(base_numpy))
            metas_to_reserve = get_istorage_attrs(base_numpy)[0].numpy_meta

        tokens = istorage_metas[0].tokens

        # Reserve array: even if we are a view we reserve space for the WHOLE numpy, as the memory

        result = cls.reserve_numpy_array(storage_id, name, metas_to_reserve) # storage_id is NOT used at all

        input_array = result[0]

        # Transform memory to a StorageNumpy
        #   This must be done BEFORE reconstructing the  view, to keep the BASE
        #   numpy loaded (otherwise the 'base' field is overwritten with the base
        #   of the view)
        if StorageNumpy._arrow_enabled(input_array):
            obj = np.asfortranarray(input_array).view(cls)
            obj._hcache_arrow = result[2]
        else: # Reserve for normal numpy
            obj = np.asarray(input_array).view(cls)

        obj._hcache = result[1]
        obj._base_metas = metas_to_reserve #Cache value to avoid cassandra accesses

        # The data recovered from the istorage is a persistent view, therefore reconstruct the view
        if getattr(istorage_metas[0], 'view_serialization', None):
            myview = pickle.loads(istorage_metas[0].view_serialization)
        else: # Accept the case where view_serialization is None, mainly due to C++ interface
            myview = tuple([slice(None,None,None)]*obj.ndim)
        log.debug(" view of {}".format(myview))
        if isinstance(myview, tuple):
            obj = super(StorageNumpy, obj).__getitem__(myview)
        else:
            raise TypeError(" WARNING: recovered 'view_serialization' has unexpected type ", type(myview))


        super().__init__(obj, name=name, storage_id=storage_id, tokens=tokens)

        obj._numpy_full_loaded = False
        obj._hcache = result[1]


        obj._build_args = obj.args(obj.storage_id, istorage_metas[0].class_name,
                istorage_metas[0].name, my_metas, istorage_metas[0].block_id, base_numpy,
                myview,
                istorage_metas[0].tokens)
        obj._row_elem = obj._hcache.get_elements_per_row(storage_id, metas_to_reserve)
        obj._calculate_nblocks(myview)
        return obj

    @classmethod
    def _parse_comments(cls, comments):
        parser = Parser("StreamOnly")
        return parser._parse_comments(comments)

    @staticmethod
    def _arrow_enabled(input_array):
        return (config.arrow_enabled and getattr(input_array, 'ndim', 0) == 2)

    def __new__(cls, input_array=None, name=None, storage_id=None, block_id=None, **kwargs):
        log.debug("input_array=%s name=%s storage_id=%s ENTER ",input_array is not None, name, storage_id)

        if input_array is not None and not isinstance(input_array, np.ndarray):
            raise AttributeError("The 'input_array' must be a numpy.ndarray instance.")

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
                log.debug(" NEW from %s", storage_id)
                # StorageNumpy(Snumpy, None, None)
                # StorageNumpy(Snumpy, name, None)
                # StorageNumpy(Snumpy, None, UUID)
                # StorageNumpy(Snumpy, name, UUID)
                if storage_id is not None:
                    log.warn("Creating a StorageNumpy with a specific StorageID")
                obj = input_array.copy()
            else:
                # StorageNumpy(numpy, None, None)
                if not StorageNumpy._arrow_enabled(input_array):
                    if StorageNumpy.USE_FORTRAN_ACCESS:
                        obj = np.asfortranarray(input_array.copy()).view(cls) #to set the fortran contiguous flag it is necessary to do the copy before
                    else:
                        obj = np.asarray(input_array).copy().view(cls)
                else:
                    obj = np.asfortranarray(input_array.copy()).view(cls) #to set the fortran contiguous flag it is necessary to do the copy before
                    log.debug("Created ARROW")
            super(StorageNumpy, obj).__init__(name=name, storage_id=storage_id, kwargs=kwargs)

            if name or storage_id: # The object needs to be persisted
                load_data= (input_array is None) and (config.load_on_demand == False)
                if input_array is not None:
                    if isinstance(input_array,StorageNumpy):
                        log.warn("Creating a Persistent StorageNumpy.")
                    obj._persist_data(obj._get_name())
                if load_data: #FIXME aixo hauria d'afectar a l'objecte existent (aqui ja existeix a memoria... o hauria)
                    obj[:]	# HACK! Load ALL elements in memory NOW (recursively calls getitem)

        if getattr(obj, "__doc__", None) is not None:
            obj._persistent_props = StorageNumpy._parse_comments(obj.__doc__)
            obj._stream_enabled = obj._persistent_props.get('stream', False)
        #print("JJ name = ", name, flush=True)
        #print("JJ _name = ", obj._name, flush=True)
        log.debug("CREATED NEW StorageNumpy storage_id=%s with input_array=%s name=%s ", storage_id, input_array is not None, name)
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

    def calculate_list_of_ranges_of_block_coords(self, view):
        """
            Return a list with the ranges of block coordinates for each dimension of 'view'.
            The block coordinates are relative to 'self.base.shape' (big).
        """
        first=[]
        last=[]
        shape = self._get_base_array().shape
        SIZE= self._row_elem # 7 # ROW_ELEM
        for idx, i in enumerate(view):
            #print(" {}:  element ={} ".format(idx, i))
            if isinstance(i, int):
                self._check_value_in_shape(i, shape[idx], idx)
                first.append(i//SIZE)
                last.append(i//SIZE)
            else: # It's a slice
                n = StorageNumpy.removenones(i, shape[idx])
                #print(" {}:    n ={} ".format(idx, n))
                #self._check_value_in_shape(n.start, shape[idx], idx) Don't fail here, allow the numpy error...
                self._check_value_in_shape(n.stop-1, shape[idx], idx)
                first.append(n.start//SIZE)
                last.append((n.stop-1)//SIZE)
        #print(" calculate_block_coords: first ={} last = {}".format(first,last), flush=True)
        l=[]
        for i in range(len(view)):
            l.append( range(first[i], last[i]+1))
        #print(" calculate_block_coords: l = {}".format(l), flush=True)
        return l

    def calculate_block_coords(self, view):
        """
        Return a list with all the block coordinates relative to 'self.base.shape' corresponding to the elements in 'view'
        """
        l = self.calculate_list_of_ranges_of_block_coords(view)

        return [b for b in itertools.product(*l)]

    @staticmethod
    def _compose_index(s, pos):
        """
        Returns the corresponding index in the slice 's' for argument 'pos'
            (2,10,2), 1 --> 4

            0  1  2  3  <------------------+
            2  4  8  10 <=== slice indexes  \
            -4 -3 -2 -1 <--------------------+- pos

        It works for negative values (...until a new case is found).
        """
        if pos < 0:
            res = s.stop  + pos*s.step
        else:
            res = s.start + pos*s.step
        return res

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
                res = old0 # 'new' is IGNORED
            elif isinstance(old0, slice):
                old0 = StorageNumpy.removenones(old0, shape[0])
                res = StorageNumpy._compose_index(old0, new)
            else:
                raise NotImplementedError("Compose an int and a {}".format(type(old0)))

        elif isinstance(new, slice):
            if isinstance(old0, int):
                res = old0  # 'new' is IGNORED
            elif isinstance(old0, slice):
                old0 = StorageNumpy.removenones(old0, shape[0])
                new = StorageNumpy.removenones(new, shape[0])
                newstart = StorageNumpy._compose_index(old0, new.start)
                newstop  = StorageNumpy._compose_index(old0, new.stop)

                if old0.step >= 0 and new.step >= 0:
                    resstep = old0.step * new.step
                else:
                    raise NotImplementedError("slice with negative steps") # TODO
                res = slice(newstart, min(newstop,old0.stop), resstep)
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
        log.debug(" view_composer: shape={} old={} new={}".format(self._get_base_array().shape, self._build_args.view_serialization, new_view))
        if isinstance(new_view, int) or isinstance(new_view,slice):
            new_view=(new_view,)
        elif not isinstance (new_view,tuple):
            raise TypeError("View must be a tuple,int or slice instead of {}".format(type(new_view)))

        old = self._build_args.view_serialization
        res = StorageNumpy.view_composer_internal(self._get_base_array().shape, old, new_view)
        log.debug(" view_composer: ======> {}".format(res))
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
        self._calculate_nblocks(new_view_serialization)
        self._persistance_needed = True

    # used as copy constructor
    def __array_finalize__(self, obj):
        if obj is None:
            log.debug("  __array_finalize__ NEW self.class={}".format(self.__class__))
            return
        log.debug("__array_finalize__ self.base=None?%s obj.base=None?%s", getattr(self, 'base', None) is None, getattr(obj, 'base', None) is None)
        if self.base is not None: # It is a view, therefore, copy data from object
            log.debug("  __array_finalize__ view (new_from_template/view)")

            self.storage_id = getattr(obj, 'storage_id', None)
            self._name = getattr(obj, '_name', None)
            self._base_metas = getattr(obj, '_base_metas', None)
            self._hcache = getattr(obj, '_hcache', None)
            if StorageNumpy._arrow_enabled(self._get_base_array()):
                self._hcache_arrow = getattr(obj, '_hcache_arrow', None)
            self._row_elem = getattr(obj, '_row_elem', None)
            # if we are a view we have ALREADY loaded all the subarray
            self._loaded_coordinates = getattr(obj, '_loaded_coordinates', [])
            self._loaded_columns = getattr(obj, '_loaded_columns', set())
            self._is_persistent = getattr(obj, '_is_persistent', False)
            self._block_id = getattr(obj, '_block_id', None)
            self._class_name = self.__class__.__module__ + '.' + self.__class__.__name__ # Put a name like 'hecuba.hnumpy.StorageNumpy'
            self._tokens = getattr(obj,'_tokens',None)
            self._build_args = getattr(obj, '_build_args', None)
            self._persistance_needed = getattr(obj, '_persistance_needed', False)
            self._persistent_columnar = getattr(obj, '_persistent_columnar', False)
            self._numpy_full_loaded = getattr(obj, '_numpy_full_loaded', False)

            if isinstance(obj, StorageNumpy): # Instantiate or getitem
                log.debug("  array_finalize obj == StorageNumpy")

                if getattr(obj, '_last_sliced_coord', None):    #getitem or split
                    if obj.shape == self.shape:
                        self._n_blocks = getattr(obj, '_n_blocks', None)
                    else:
                        log.debug("  array_finalize obj.shape != self.shape create persistent view")
                        self._create_lazy_persistent_view(obj._last_sliced_coord)

                    if self.is_columnar(self._build_args.view_serialization):
                        self._persistent_columnar = True

                    obj._last_sliced_coord = None
                    self._numpy_full_loaded = True # By default assume we come from a getitem, otherwise mark it as appropiate (split)

            else:
                # StorageNumpy from a numpy
                log.debug("  array_finalize obj != StorageNumpy")
                self._numpy_full_loaded = True # Default value
        else:
            log.debug("  __array_finalize__ copy")
            # Initialize fields as the __new__ case with input_array and not name
            self._loaded_coordinates = []
            self._loaded_columns     = set()
            self._numpy_full_loaded  = True # FIXME we only support copy for already loaded objects
            self._name               = None
            self.storage_id          = None
            self._is_persistent      = False
            self._class_name         = self.__class__.__module__ + '.' + self.__class__.__name__ #name as 'hecuba.hnumpy.StorageNumpy'
            self._block_id           = getattr(obj, '_block_id', None)
            self._persistance_needed = False
            self._persistent_columnar= False


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
            if StorageNumpy._arrow_enabled(result[0]):
                hcache_arrow = StorageNumpy._create_hcache(StorageNumpy.get_arrow_name(name))
                return [result[0], hcache, hcache_arrow]
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
        load = True # By default, load everything
        if new_coords is None: # Special case: Load everything
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
            base_numpy = self._get_base_array()
            metas = self._base_metas
            log.debug("  COORDINATES ARE {} ".format(new_coords))
            self._hcache.load_numpy_slices([self._build_args.base_numpy], metas, [base_numpy],
                                   new_coords,
                                   StorageNumpy.BLOCK_MODE)

    def is_columnar(self,sliced_coord):
        if not StorageNumpy._arrow_enabled(self._get_base_array()):
            log.debug("HECUBA_ARROW is not enabled or dimensions > 2. Columnar acces disabled.")
            return False
        if self._persistent_columnar == True:
            return True
        # if the number of rows is very low we do not use columnar access
        if self.shape[0]<50:
            #print("self.shape[0]<50", flush=True)
            return False

        if isinstance(sliced_coord, slice) and (sliced_coord == slice(None, None, None) or sliced_coord == slice(0, self._get_base_array().shape[0],1)):
            return True
        if isinstance(sliced_coord, tuple):
            # If the getitem parameter is a tuple, then we may catch the
            # column accesses: Ex: s[:, i], s[:, [i1,i2]], s[:, slice(...)]
            # All these accesses arrive here as a tuple:
            #   (slice(None,None,None), xxx)
            #   or a slice that has ALL elements
            # where xxx is the last parameter of the tuple.
            # FIXME Extend to more than 2 dimensions
            dims = sliced_coord.__len__()
            if dims == 2: # Only 2 dimensions
                if isinstance(sliced_coord[-dims], slice) and (sliced_coord[-dims] == slice(None, None, None) or sliced_coord[-dims]==slice(0,self._get_base_array().shape[-dims],1)):
                    return True
            return False
        return False

    def _select_columns(self, sliced_coord):
        """
        Returns None or a list of columns accesed by 'sliced_coord'
            The list of columns is calculated on top of 'self.base'
        """
        columns = None
        last = sliced_coord[-1]
        if isinstance (last,int):
            columns = [last]
        else: # it is an slice
            last = StorageNumpy.removenones(last, self._get_base_array().shape[1])
            columns = [ c for c in range(last.start, last.stop, last.step)]

        log.debug(" _select_columns ({}) ==> {}".format(sliced_coord, columns))
        return columns

    def _check_value_in_shape(self, value, shape, axis):
        if (value < 0) or (value > shape):
            raise IndexError("index {} is out of bounds for axis {} with size {}".format(value, axis, shape))

    def _check_columns_in_bounds(self, columns):
        """
            Check that the list of columns belongs to shape in base, or raise an exception
        """
        for col in columns:
            self._check_value_in_shape(col, self._get_base_array().shape[1], 1)

    def _load_columns(self, columns):
        """
            Load from Cassandra the list of columns.
            Args:
                self: The StorageNumpy to load data into
                columns: The coordinates to load (column position)
            PRE: self._is_persistent and not self._numpy_full_loaded
        """

        self._check_columns_in_bounds(columns)
        load = True
        coordinates = self._loaded_columns.union(columns)
        if (len(coordinates) != len(self._loaded_columns)):
            self._numpy_full_loaded = (len(coordinates) == self.shape[1])
            self._loaded_columns = coordinates
            if not self._persistent_columnar:
                log.debug("_load_columns: Enabling columnar access %s", self.storage_id)
                self._persistent_columnar = True
        else:
            load = False
        if load:
            log.debug("LOADING COLUMNS {}".format(columns))
            base_numpy = self._get_base_array()
            self._hcache_arrow.load_numpy_slices([self._build_args.base_numpy],
                                    self._base_metas,
                                    [base_numpy],
                                    columns,
                                    StorageNumpy.COLUMN_MODE)

    def _select_and_load_blocks(self, sliced_coord):
        """
            PRE: self._is_persistent and not self._numpy_full_loaded
        """
        block_coord = self._select_blocks(sliced_coord)
        self._load_blocks(block_coord)

    def _references_single_element(self, sliced_coord):
        '''
        Returns True if the 'sliced_coord' references a single element from 'self'
        '''
        if isinstance(sliced_coord,tuple):
            if len(sliced_coord) != self.ndim:
                return False
            for i in range(len(sliced_coord)):
                if not isinstance(sliced_coord[i],int):
                    return False
            return True
        if isinstance(sliced_coord, int):
            return self.ndim == 1
        return False

    def __getitem__(self, sliced_coord):
        log.info("RETRIEVING NUMPY {} is_persistent {}".format(sliced_coord, self._is_persistent))
        if self._is_persistent:
            if not (self._numpy_full_loaded and self._references_single_element(sliced_coord)): # Optimization to avoid 'view_composer' for single accessess

                #if the slice is a npndarray numpy creates a copy and we do the same
                if isinstance(sliced_coord, np.ndarray): # is there any other slicing case that needs a copy of the array????
                    result = self.view(np.ndarray)[sliced_coord] # TODO: If self is NOT loaded LOAD IT ALL BEFORE
                    return self.__class__(result) # Creates a copy (A StorageNumpy from a Numpy)

                self._last_sliced_coord = sliced_coord  # Remember the last getitem parameter, because it may force a new entry in the istorage at array_finalize

                if not self._numpy_full_loaded:
                    # Use 'big_sliced_coord' to access disk and 'sliced_coord' to access memory
                    # Keep 'sliced_coord' to reuse the common return at the end
                    big_sliced_coord = self._view_composer_new(sliced_coord)
                    if self.is_columnar(big_sliced_coord):
                        columns = self._select_columns(big_sliced_coord)
                        if columns is not None : # Columnar access
                            self._load_columns(columns)
                    else: # Normal array access...
                        self._select_and_load_blocks(big_sliced_coord)
        return super(StorageNumpy, self).__getitem__(sliced_coord)

    def __setitem__(self, sliced_coord, values):
        log.info("WRITING NUMPY")
        log.debug("setitem %s", sliced_coord)
        if isinstance(values, StorageNumpy) and values._is_persistent and not values._numpy_full_loaded:
            values[:]  # LOAD the values as the numpy.__setitem__ will only use memory
        if self._is_persistent:
            big_sliced_coord = self._view_composer_new(sliced_coord)
            block_coords = self._select_blocks(big_sliced_coord)
            if not self._numpy_full_loaded: # Load the block before writing!
                self._load_blocks(block_coords)

            #yolandab: execute first the super to modified the base numpy
            super(StorageNumpy, self).__setitem__(sliced_coord, values)

            base_numpy = self._get_base_array() # self.base is  numpy.ndarray
            metas = self._base_metas
            self._hcache.store_numpy_slices([self._build_args.base_numpy],
                    metas, [base_numpy],
                    block_coords,
                    StorageNumpy.BLOCK_MODE)
            return
        super(StorageNumpy, self).__setitem__(sliced_coord, values)
        return

    def _initialize_stream_capability(self, topic_name=None):
        super()._initialize_stream_capability(topic_name)
        if topic_name is not None:
            self._hcache.enable_stream(self._topic_name, {'kafka_names': str.join(",",config.kafka_names)})

    def send(self, key=None, val=None):
        """
        Send to KAFKA the key (ignored) and value (metas + numpyblocks)
        """
        if not self._is_persistent:
            raise RuntimeError("'send' operation is only valid on persistent objects")
        if not self._is_stream() :
            raise RuntimeError("current NUMPY {} is not a stream".format(self._name))

        if getattr(self,"_topic_name",None) is None:
            self._initialize_stream_capability(self.storage_id)

        if not self._stream_producer_enabled:
            self._hcache.enable_stream_producer()
            self._stream_producer_enabled = True

        if key is None and val is None:
            # Send the WHOLE numpy
            self._hcache.send_event(self._build_args.metas, [self._get_base_array()], None)

        else:
            raise NotImplementedError("SEND partial numpy not supported")

    def poll(self):
        log.debug("StorageNumpy: POLL ")

        if not self._is_stream():
            raise RuntimeError("Poll on a not streaming object")

        if getattr(self,"_topic_name",None) is None:
            self._initialize_stream_capability(self.storage_id)

        if not self._stream_consumer_enabled:
            self._hcache.enable_stream_consumer()
            self._stream_consumer_enabled=True

        self._hcache.poll(self._build_args.metas, [self._get_base_array()])
        self._numpy_full_loaded = True
        return self

    def _persist_data(self, name, formato=0):
        """
        Persist data to cassandra, the common attributes have been generated by IStorage.make_persistent
        Args:
            StorageNumpy to persist
            name to use
            [formato] to store the data (0-ZOrder, 2-columnar, 3-FortranOrder) # 0 ==Z_ORDER (find it at SpaceFillingCurve.h)
        """
        log.debug("_persist_data: {} format={} ENTER ".format(name, formato))

        if None in self or not self.ndim:
            raise NotImplemented("Empty array persistance")

        if not getattr(self,'_built_remotely', None):
            if StorageNumpy._arrow_enabled(self._get_base_array()):
                if formato == 0: # If arrow & ZORDER -> FortranOrder
                    formato = 3
                self._create_tables_arrow(StorageNumpy.get_arrow_name(name))
            if StorageNumpy.USE_FORTRAN_ACCESS:
                formato = 3
            self._create_tables(name)

        if not getattr(self, '_hcache', None):
            if StorageNumpy._arrow_enabled(self._get_base_array()):
                self._hcache_arrow = self._create_hcache(StorageNumpy.get_arrow_name(name))
            self._hcache = self._create_hcache(name)

        log.debug("_persist_data: after create tables and cache ")
        # Persist current object
        hfetch_metas = HArrayMetadata(list(self.shape), list(self.strides),
                                      self.dtype.kind, self.dtype.byteorder,
                                      self.itemsize, self.flags.num, formato)
        self._base_metas = hfetch_metas
        self._build_args = self.args(self.storage_id, self._class_name, self._get_name(), hfetch_metas, self._block_id,
                                     self.storage_id, # base_numpy is storage_id because until now we only reach this point if we are not inheriting from a StorageNumpy. We should update this if we allow StorageNumpy from volatile StorageNumpy
                                     tuple([slice(None,None,None)]*self.ndim),  #We are a view of everything (view_serialization)
                                     self._tokens)
        if len(self.shape) != 0:
            sid = self._build_args.base_numpy

            log.debug("_persist_data: before store slices ROW")
            if self.shape != self._get_base_array().shape:
                raise NotImplementedError("Persisting a volatile view with different shape is NOT implemented")
            self._hcache.store_numpy_slices([sid], self._build_args.metas, [self._get_base_array()], # CHECK metas del padre i memoria tienen que coincidir
                                            None,
                                            StorageNumpy.BLOCK_MODE)
            log.debug("_persist_data: before store slices COLUMN")
            if StorageNumpy._arrow_enabled(self._get_base_array()):
                self._hcache_arrow.store_numpy_slices([sid], self._build_args.metas, [self._get_base_array()], # CHECK metas del padre i memoria tienen que coincidir
                                                None,
                                                StorageNumpy.COLUMN_MODE)
            self._row_elem = self._hcache.get_elements_per_row(sid, self._build_args.metas)
            self._calculate_nblocks(self._build_args.view_serialization)
        log.debug("_persist_data: before store meta")
        StorageNumpy._store_meta(self._build_args)
        log.debug("_persist_data: before get_elements_per_row")
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
        self.sync() # TODO: we should discard pending writes
        super().delete_persistent()

        query = "DROP TABLE %s;" %(self._get_name())
        query2 = "DELETE FROM hecuba.istorage WHERE storage_id = %s;" % self.storage_id
        log.debug("DELETE PERSISTENT: %s", query)
        config.session.execute(query)
        config.session.execute(query2)
        self.storage_id = None

    def sync(self):
        """
        Wait until all pending stores to Cassandra have been finished.
        """
        log.debug("SYNC: %s", self.storage_id)
        self._hcache.wait()

    def __iter__(self):
        if self._numpy_full_loaded:
            return iter(self.view(np.ndarray))
        else:
            return iter(self[:].view(np.ndarray))

    def __contains__(self, item):
        return item in self.view(np.ndarray)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        log.debug(" UFUNC method({}) ".format(method))
        log.debug(" UFUNC self sid ({}) ".format(getattr(self,'storage_id',None)))
        args = []
        for input_ in inputs:
            log.debug(" UFUNC input loop sid={}".format(getattr(input_,'storage_id',None)))
            if isinstance(input_, StorageNumpy):
                StorageNumpy._preload_memory(input_)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = kwargs.pop('out', None)
        if outputs:
            out_args = []
            for output in outputs:
                log.debug(" UFUNC output loop sid={}".format(getattr(output,'storage_id',None)))
                if isinstance(output, StorageNumpy):
                    StorageNumpy._preload_memory(output)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        base_numpy = self._get_base_array()
        if self._is_persistent and len(self.shape) and self._numpy_full_loaded is False:
            StorageNumpy._preload_memory(self)
            #metas = self._base_metas
            #log.debug(" UFUNC({}) load_block from {} ".format(method, metas))
            #if StorageNumpy._arrow_enabled(base_numpy):
            #    load_method = StorageNumpy.COLUMN_MODE
            #    self._hcache_arrow.load_numpy_slices([self._build_args.base_numpy], metas, [base_numpy],
            #                               None,
            #                               load_method)
            #else:
            #    load_method = StorageNumpy.BLOCK_MODE
            #    self._hcache.load_numpy_slices([self._build_args.base_numpy], metas, [base_numpy],
            #                               None,
            #                               load_method)
        results = super(StorageNumpy, self).__array_ufunc__(ufunc, method,
                                                            *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented
        log.debug(" UFUNC: type(results)=%s results is self? %s outputs[0] is results? %s outputs[0] is self? %s", type(results), results is self, outputs[0] is results, outputs[0] is self)
        if method == 'at':
            return

        if self._is_persistent and len(self.shape):
            readonly_methods = ['mean', 'sum', 'reduce'] #methods that DO NOT modify the original memory, and there is NO NEED to store it
            if method not in readonly_methods:
                if self in outputs: # Self must store the value
                    block_coord = self._select_blocks(self._build_args.view_serialization)
                    self._hcache.store_numpy_slices([self._build_args.base_numpy], self._base_metas, [base_numpy],
                                                block_coord,
                                                StorageNumpy.BLOCK_MODE)

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((result
                         if output is None else output)
                        for result, output in zip(results, outputs))

        return results[0] if len(results) == 1 else results


    def reshape(self, newshape, order=None):
        '''
        reshape the StorageNumpy
        Creates a view of the StorageNumpy sharing data with the original data (Both in disk and memory)
        '''
        log.debug("reshape from %s to %s", self.shape, newshape)
        if order == None:
            if is_columnar(self):
                order = 'A'
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
    @staticmethod
    def _preload_memory(a):
        """
            Load a persistent object in memory.
        """
        srcA = a
        if isinstance(a, StorageNumpy) and a._is_persistent and not a._numpy_full_loaded:
            log.debug(" PRELOAD: sid = {} ".format(a.storage_id))
            srcA = a[:]	# HACK! Load ALL elements in memory NOW (recursively calls getitem)
        return srcA

    def dot(a, b, out=None):
        srcA = StorageNumpy._preload_memory(a)
        srcB = StorageNumpy._preload_memory(b)

        log.debug(" DOT: AFTER PRELOAD ")
        return config.intercepted['dot'](srcA,srcB,out) # At the end of this 'copy' is called

    def array_equal(a, b):
        srcA = StorageNumpy._preload_memory(a)
        srcB = StorageNumpy._preload_memory(b)
        log.debug(" array_equal: AFTER PRELOAD ")
        return config.intercepted['array_equal'](srcA,srcB)

    def concatenate(sn_list,axis=0, out=None):
        preloaded_sn=[]
        for i in range(len(sn_list)):
             preloaded_sn.append(StorageNumpy._preload_memory(sn_list[i]))
        log.debug(" concatenate: AFTER PRELOAD ")
        return config.intercepted['concatenate'](preloaded_sn,axis, out)

