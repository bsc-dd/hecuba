# author: G. Alomar

import time
from collections import defaultdict

from hecuba import config


class Block(object):
    """
    Object used to access data from workers.
    """

    @staticmethod
    def build_remotely(results):
        """
        Launches the Block.__init__ from the api.getByID
        Args:
            results: a list of all information needed to create again the block
        """
        return Block(results.blockid, results.entry_point, results.dict_name, results.tab, results.ksp, results.tkns,
                     results.storageobj_classname)

    def __init__(self, blockid, peer, keynames, tablename, keyspace, tokens, storageobj_classname):
        """
        Creates a new block.

        Args:
            blockid (string):  an unique block identifier
            peer (string): hostname
            keynames (list): the Cassandra partition key
            tablename (string): the name of the collection/table
            keyspace (string): name of the Cassandra keyspace.
            tokens (list): list of tokens
            storageobj_classname (string): full class name of the storageobj

        Returns:

        """
        self.blockid = blockid
        self.node = peer
        self.token_ranges = tokens
        self.key_names = keynames
        self.table_name = tablename
        if type(keynames) is not list:
            raise TypeError
        self.keyspace = keyspace
        self.needContext = True
        self.supportsPrefetch = True
        self.supportsStatistics = False
        last = 0
        for key, i in enumerate(storageobj_classname):
            if i == '.' and key > last:
                last = key
        module = storageobj_classname[:last]
        cname = storageobj_classname[last + 1:]
        mod = __import__(module, globals(), locals(), [cname], 0)
        self.storageobj = getattr(mod, cname)(keyspace+"."+tablename)
        self.cntxt = ""

    def __iter__(self):
        return BlockIter(self)

    def __getitem__(self, key):
        """
            Launches the getitem of the dict found in the block storageobj
            Args:
                key: the position of the value that we're looking for
            Returns:
                val: the value that we're looking for
        """
        return self.storageobj[key]

    def __setitem__(self, key, val):
        """
           Launches the setitem of the dict found in the block storageobj
           Args:
               key: the position of the value that we want to save
               val: the value that we want to save in that position
           Returns:
        """
        self.storageobj[key] = val

    def getID(self):
        """
        Obtains the id of the block
        Returns:
            self.blockid: id of the block
        """
        return self.blockid

    def __iter__(self):
        """
        Obtains the iterator for the keys of the block
        Returns:
            BlockIter(self): list of keys
        """
        return BlockIter(self)

    def iteritems(self):
        """
        Obtains the iterator for the key,val pairs of the block
        Returns:
            BlockItemsIter(self): list of key,val pairs
        """
        return BlockItemsIter(self)

    def itervalues(self):
        """
        Obtains the iterator for the values of the block
        Returns:
            BlockValuesIter(self): list of values
        """
        return BlockValuesIter(self)

    def iterkeys(self):
        """
        Obtains the iterator for the keys of the block
        Returns:
            iterkeys(self): list of keys
        """
        return BlockIter(self)


class BlockIter(object):
    """
    Iterator for the keys of the block
    """

    def __iter__(self):
        """
        Needed to be considered as an iterable
        """
        return self

    def __init__(self, iterable):
        """
        Initializes the iterator, and its prefetcher
        Args:
            iterable: Block to iterate over
        """
        self.pos = 0
        self.keys = []
        self.num_keys = 0
        self.iterable = iterable
        self.end = False

        keys = self.iterable.storageobj.keyList[self.iterable.storageobj.__class__.__name__]
        exec ("self.iterable.storageobj." + str(keys[0]) + ".prefetchManager.pipeq_write[0].send(['query'])")

    def next(self):
        """
        Returns the keys, one by one, contained in the token ranges of the block
        Returns:
            key: .
        """
        if self.pos == self.num_keys:
            if self.end:
                raise StopIteration
            else:
                keys = self.iterable.storageobj.keyList[self.iterable.storageobj.__class__.__name__]
                exec ("self.iterable.storageobj." + str(keys[0]) + ".prefetchManager.pipeq_write[0].send(['continue'])")
                exec ("usedpipe = self.iterable.storageobj." + str(keys[0]) + ".prefetchManager.piper_read[0]")
                results = usedpipe.recv()
                if len(results) == 0:
                    raise StopIteration
                else:
                    self.keys = []
                    exec ("currpersistentdict = self.iterable.storageobj." + str(keys[0]))
                    for entry in results:
                        currpersistentdict.dictCache[entry[0]] = [entry[1], 'Sync']
                        self.keys.append(entry[0])
                    self.num_keys = len(self.keys)
                    self.pos = 0
                    if len(results) < 100:
                        self.end = True

        if len(self.keys) == 0:
            print "Error obtaining block_keys in iter.py"

        key = self.keys[self.pos]
        self.pos += 1
        return key


class BlockItemsIter(object):
    """
        Iterator for the key,val pairs of the block
    """

    def __iter__(self):
        """
        Needed to be considered as an iterable
        """
        return self

    def __init__(self, iterable):
        """
        Initializes the iterator, and its prefetcher
        Args:
            iterable: Block to iterate over
        """
        self.pos = 0
        self.keys = []
        self.num_keys = 0
        self.iterable = iterable
        self.end = False
        self.persistentDict = ""

        keys = self.iterable.storageobj.keyList[self.iterable.storageobj.__class__.__name__]
        exec ("self.persistentDict = self.iterable.storageobj." + str(keys[0]))
        self.persistentDict.prefetchManager.pipeq_write[0].send(['query'])

    def next(self):
        """
        Returns the keys,value pairs, one by one, contained in the token ranges of the block
        Returns:
            (key,val): .
        """
        if self.pos == self.num_keys:
            if self.end:
                raise StopIteration
            else:
                self.persistentDict.prefetchManager.pipeq_write[0].send(['continue'])
                usedpipe = self.persistentDict.prefetchManager.piper_read[0]
                results = usedpipe.recv()
                if len(results) == 0:
                    raise StopIteration
                else:
                    self.keys = []
                    for entry in results:
                        self.persistentDict.dictCache[entry[0]] = [entry[1], 'Sync']
                        self.keys.append(entry[0])
                    self.num_keys = len(self.keys)
                    self.pos = 0
                    if len(results) < 100:
                        self.end = True

        if len(self.keys) == 0:
            print "Error obtaining block_keys in iter.py"

        key = self.keys[self.pos]
        value = self.persistentDict.dictCache[key]
        self.pos += 1
        return key, value[0]


class BlockValuesIter(object):
    """
        Iterator for the values of the block
    """

    def __iter__(self):
        """
        Needed to be considered as an iterable
        """
        return self

    def __init__(self, iterable):
        """
        Initializes the iterator, and its prefetcher
        Args:
            iterable: Block to iterate over
        """
        self.pos = 0
        self.keys = []
        self.num_keys = 0
        self.iterable = iterable
        self.end = False
        self.persistentDict = ""

        keys = self.iterable.storageobj.keyList[self.iterable.storageobj.__class__.__name__]
        exec ("self.persistentDict = self.iterable.storageobj." + str(keys[0]))
        self.persistentDict.prefetchManager.pipeq_write[0].send(['query'])

    def next(self):
        """
        Returns the values, one by one, contained in the token ranges of the block
        Returns:
            val: .
        """
        self.persistentDict.reads += 1  # STATISTICS
        if self.pos == self.num_keys:
            if self.end:
                raise StopIteration
            else:
                self.persistentDict.prefetchManager.pipeq_write[0].send(['continue'])
                usedpipe = self.persistentDict.prefetchManager.piper_read[0]
                self.persistentDict.cache_prefetchs += 1  # STATISTICS
                self.persistentDict.cache_hits_graph += 'P'  # STATISTICS
                start = time.time()  # STATISTICS
                results = usedpipe.recv()
                self.persistentDict.pending_requests_time += time.time() - start  # STATISTICS
                if len(results) == 0:
                    raise StopIteration
                else:
                    self.keys = []
                    for entry in results:
                        self.persistentDict.cache_hits += 1  # STATISTICS
                        self.persistentDict.cache_hits_graph += 'X'  # STATISTICS
                        start = time.time()  # STATISTICS
                        self.persistentDict.dictCache[entry[0]] = [entry[1], 'Sync']
                        self.persistentDict.cache_hits_time += time.time() - start  # STATISTICS
                        self.keys.append(entry[0])
                    self.num_keys = len(self.keys)
                    self.pos = 0
                    if len(results) < 100:
                        self.end = True

        if len(self.keys) == 0:
            self.persistentDict.miss += 1  # STATISTICS
            self.persistentDict.cache_hits_graph += '_'  # STATISTICS
            print "Error obtaining block_keys in iter.py"

        value = self.persistentDict.dictCache[self.keys[self.pos]]
        self.pos += 1
        return value[0]


class KeyIter(object):
    """
        Iterator for the blocks of the storageobj
    """
    blockKeySpace = ''

    def __init__(self, keyspace, table, storage_class, primary_keys):
        """
        Initializes the iterator, and saves the information about the token ranges of each block

        Args:
            iterable (hecuba.storageobj.StorageObj): Block to iterate over

        """
        self.pos = 0
        self.ring = []
        self.n_blocks = config.number_of_blocks
        self._keyspace = keyspace
        self._table = table
        self._storage_class = storage_class
        self._primary_keys = primary_keys
        metadata = config.cluster.metadata
        token_to_hosts = map(lambda (tkn, host): (tkn.value, host.address),
                             metadata.token_map.token_to_host_owner.iteritems())
        self.ring = KeyIter._calulate_block_ranges(token_to_hosts, config.number_of_blocks)

    @staticmethod
    def _calulate_block_ranges(token_to_host, n_blocks):
        host_to_tokens = defaultdict(list)
        for t, h in token_to_host.iteritems():
            host_to_tokens[h].append((t, h))

        ring = []
        for tokens in host_to_tokens.values():
            ring += tokens
        tks = []
        n_tokens = len(token_to_host)
        if n_tokens < n_blocks:
            raise ValueError('Use virtual tokens!')
        elif n_tokens % n_blocks == 0:
            token_per_block = n_tokens / n_blocks
            tks = [[] for i in range(n_blocks)]
            for i in range(n_tokens):
                tks[i / token_per_block].append(ring[i])

        return tks

    def __iter__(self):
        return self

    def next(self):
        """
        Returns the blocks, one by one, created from the data in the storageobj
        Returns:
            block (Block): a block representing the partition of the dictionary.
        """
        if self.n_blocks == self.pos:
            raise StopIteration

        current_pos = self.ring[self.pos]  # [1]
        host = current_pos[0][1]
        tks = map(lambda a: a[0], current_pos)
        import uuid
        myuuid = str(uuid.uuid1())

        config.session.execute(
            'INSERT INTO hecuba.blocks (blockid, block_classname,storageobj_classname,tkns, ksp, tab, obj_type, entry_point)' +
            ' VALUES (%s,%s,%s,%s,%s,%s,%s,%s)',
            [myuuid, "hecuba.iter.Block", self._storage_class, tks, self._keyspace, self._table, 'hecuba', host])


        b = Block(myuuid, host, self._primary_keys, self._table, self._keyspace, tks, self._storage_class)
        self.pos += 1
        return b
