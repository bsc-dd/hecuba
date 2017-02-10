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
        return Block(results.blockid.encode('utf8'), results.entry_point.encode('utf8'), results.tab.encode('utf8'),
                     results.ksp.encode('utf8'), results.tkns, results.storageobj_classname.encode('utf8'),
                     results.key_list, results.value_list,
                     results.object_id.encode('utf8'))

    def __init__(self, blockid, peer, tablename, keyspace, tokens, storageobj_classname, primary_keys, columns, storage_obj_id=None):
        """
        Creates a new block.

        Args:
            blockid (string):  an unique block identifier
            peer (string): hostname
            tablename (string): the name of the collection/table
            keyspace (string): name of the Cassandra keyspace.
            tokens (list): list of tokens
            storageobj_classname (string): full class name of the storageobj
            storage_obj_id (string): id of the storageobj
        """
        self.blockid = blockid
        self.node = peer
        self.token_ranges = tokens
        self.table_name = tablename
        self.keyspace = keyspace
        self._needContext = True
        self.supportsPrefetch = True
        self.supportsStatistics = False
        self.keys = primary_keys
        self.values = columns
        last = 0
        for key, i in enumerate(storageobj_classname):
            if i == '.' and key > last:
                last = key
        module = storageobj_classname[:last]
        cname = storageobj_classname[last + 1:]
        mod = __import__(module, globals(), locals(), [cname], 0)
        self.storageobj = getattr(mod, cname)(keyspace + "." + tablename, myuuid=storage_obj_id)
        self.cntxt = ""

    def __eq__(self, other):
        return self.blockid == other.blockid and self.node == other.node and \
               self.token_ranges == other.token_ranges \
               and self.table_name == other.table_name and self.keyspace == other.keyspace \
               and self._needContext == other._needContext and self.supportsPrefetch == other.supportsPrefetch \
               and self.supportsStatistics == other.supportsStatistics and self.storageobj == other.storageobj \
               and self.cntxt == other.cntxt

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
            iterkeys(self): list of keys
        """
        if config.prefetch_activated:
            return BlockIterPrefetch(self.storageobj._get_default_dict())
        else:
            partition_key = self.storageobj._get_default_dict()._primary_keys[0]
            return BlockIter(self, partition_key, self.keyspace, self.table_name, self.token_ranges)

    def iterkeys(self):
        """
        Obtains the iterator for the keys of the block
        Returns:
            iterkeys(self): list of keys
        """
        if config.prefetch_activated:
            return BlockIterPrefetch(self.storageobj._get_default_dict())
        else:
            # partition_key = self.storageobj._get_default_dict()._primary_keys[0]
            partition_key = map(lambda(x,y):(x), self.storageobj._get_default_dict()._primary_keys)
            return BlockIter(self, partition_key, self.keyspace, self.table_name, self.token_ranges)

    def iteritems(self):
        """
        Obtains the iterator for the key,val pairs of the block
        Returns:
            BlockItemsIter(self): list of key,val pairs
        """
        if config.prefetch_activated:
            return BlockItemsIterPrefetch(self.storageobj._get_default_dict())
        else:
            # partition_key = self.storageobj._get_default_dict()._primary_keys[0]
            partition_key = map(lambda(x,y):(x), self.storageobj._get_default_dict()._primary_keys)
            return BlockItemsIter(self, partition_key, self.keyspace, self.table_name, self.token_ranges)

    def itervalues(self):
        """
        Obtains the iterator for the values of the block
        Returns:
            BlockValuesIter(self): list of values
        """
        if config.prefetch_activated:
            return BlockValuesIterPrefetch(self.storageobj._get_default_dict())
        else:
            # partition_key = self.storageobj._get_default_dict()._primary_keys[0]
            partition_key = map(lambda(x,y):(x), self.storageobj._get_default_dict()._primary_keys)
            return BlockValuesIter(self, partition_key, self.keyspace, self.table_name, self.token_ranges)


class BlockIter(object):
    """
    Iterator for the keys of the block
    """

    def __iter__(self):
        """
        Needed to be considered as an iterable
        """
        return self

    def __init__(self, block, partition_key, keyspace, table, block_tokens):
        """
        Initializes the iterator
        Args:
            partition_key list(String):
            keyspace (String):
            table (String):
            block_tokens (String):
        """
        self._token_pos = 0
        self.block = block
        self._query = config.session.prepare(
            "SELECT * FROM " + keyspace + "." + table + " WHERE token(" + ",".join(partition_key) + ") >= ? AND " +
            "token(" + ",".join(partition_key) + ") < ?")
        metadata = config.cluster.metadata
        ringtokens = metadata.token_map
        ran = set(block_tokens)
        last = ringtokens.ring[len(ringtokens.ring) - 1]
        self._token_ranges = []
        max_token = 0
        min_token = 0
        for t in ringtokens.ring:
            if t.value > max_token:
                max_token = t.value
            if t.value < min_token:
                min_token = t.value

        for t in ringtokens.ring:
            if t.value in ran:
                if t.value == min_token:
                    self._token_ranges.append((-9223372036854775808, min_token))
                    self._token_ranges.append((max_token, 9223372036854775807))
                else:
                    self._token_ranges.append((last.value, t.value))
            last = t

        self._current_iterator = None

    def next(self):
        """
        Returns the values, one by one, contained in the token ranges of the block
        Returns:
            val: .
        """
        if self._current_iterator is not None:
            try:
                to_return = self._current_iterator.next()
                keys = []
                for k in self.block.keys:
                    try:
                        keys.append(getattr(to_return, str(k)))
                    except Exception:
                        pass
                if len(keys) == 1:
                    keys = keys[0]
                else:
                    keys = tuple(keys)
                return keys
            except StopIteration:
                # If the current iterator is empty, we try the next token range.
                pass

        if self._token_pos < len(self._token_ranges):
            query = self._query.bind(self._token_ranges[self._token_pos])
            self._current_iterator = iter(config.session.execute(query))
            self._token_pos += 1
            return self.next()
        else:
            raise StopIteration


class BlockIterPrefetch(object):
    """
        Iterator for the keys of the block
    """

    def __iter__(self):
        """
        Needed to be considered as an iterable
        """
        return self

    def __init__(self, default_dict):
        """
        Initializes the iterator, and its prefetcher
        Args:
            default_dict: Block to iterate over
        """
        self.pos = 0
        self.keys = []
        self.num_keys = 0
        self.end = False

        self._persistentDict = default_dict
        self._persistentDict.prefetchManager.pipeq_write[0].send(['query'])

    def next(self):
        """
        Returns the values, one by one, contained in the token ranges of the block
        Returns:
            val: .
        """
        self._persistentDict.reads += 1  # STATISTICS
        if self.pos == self.num_keys:
            if self.end:
                raise StopIteration
            else:
                self._persistentDict.prefetchManager.pipeq_write[0].send(['continue'])
                usedpipe = self._persistentDict.prefetchManager.piper_read[0]
                self._persistentDict.cache_prefetchs += 1  # STATISTICS
                self._persistentDict.cache_hits_graph += 'P'  # STATISTICS
                start = time.time()  # STATISTICS
                results = usedpipe.recv()
                self._persistentDict.pending_requests_time += time.time() - start  # STATISTICS
                if len(results) == 0:
                    raise StopIteration
                else:
                    self.keys = []
                    for entry in results:
                        self._persistentDict.cache_hits += 1  # STATISTICS
                        self._persistentDict.cache_hits_graph += 'X'  # STATISTICS
                        start = time.time()  # STATISTICS
                        self._persistentDict.dictCache[entry[0]] = [entry[1], 'Sync']
                        self._persistentDict.cache_hits_time += time.time() - start  # STATISTICS
                        self.keys.append(entry[0])
                    self.num_keys = len(self.keys)
                    self.pos = 0
                    if len(results) < config.batch_size:
                        self.end = True

        if len(self.keys) == 0:
            self._persistentDict.miss += 1  # STATISTICS
            self._persistentDict.cache_hits_graph += '_'  # STATISTICS
            print "Error obtaining block_keys in iter.py"

        key = self.keys[self.pos]
        self.pos += 1
        return key


class BlockItemsIter(object):
    """
        Iterator for the key,value pairs of the block
    """

    def __iter__(self):
        """
        Needed to be considered as an iterable
        """
        return self

    def __init__(self, block, partition_key, keyspace, table, block_tokens):
        """
        Initializes the iterator
        Args:
            partition_key (String):
            keyspace (String):
            table (String):
            block_tokens (String):
        """
        self._token_pos = 0
        self.block = block
        # TODO this does not work if the primary key is composed
        self._query = config.session.prepare(
            "SELECT * FROM " + keyspace + "." + table + " WHERE token(" + ",".join(partition_key) + ") >= ? AND " +
            "token(" + ",".join(partition_key) + ") < ?")
        metadata = config.cluster.metadata
        ringtokens = metadata.token_map
        ran = set(block_tokens)
        last = ringtokens.ring[len(ringtokens.ring) - 1]
        self._token_ranges = []
        max_token = 0
        min_token = 0
        for t in ringtokens.ring:
            if t.value > max_token:
                max_token = t.value
            if t.value < min_token:
                min_token = t.value

        for t in ringtokens.ring:
            if t.value in ran:
                if t.value == min_token:
                    self._token_ranges.append((-9223372036854775808, min_token))
                    self._token_ranges.append((max_token, 9223372036854775807))
                else:
                    self._token_ranges.append((last.value, t.value))
            last = t

        self._current_iterator = None

    def next(self):
        """
        Returns the values, one by one, contained in the token ranges of the block
        Returns:
            val: .
        """

        if self._current_iterator is not None:
            try:
                to_return = self._current_iterator.next()
                keys = []
                for k in self.block.keys:
                    try:
                        keys.append(getattr(to_return, str(k.encode('utf8'))))
                    except Exception:
                        pass
                if len(keys) == 1:
                    keys = keys[0]
                else:
                    keys = tuple(keys)
                values = []
                for v in self.block.values:
                    try:
                        values.append(getattr(to_return, str(v.encode('utf8'))))
                    except Exception:
                        pass
                if len(values) == 1:
                    values = values[0]
                else:
                    values = tuple(values)
                return keys, values
            except StopIteration:
                # If the current iterator is empty, we try the next token range.
                pass

        if self._token_pos < len(self._token_ranges):
            query = self._query.bind(self._token_ranges[self._token_pos])
            self._current_iterator = iter(config.session.execute(query))
            self._token_pos += 1
            return self.next()
        else:
            raise StopIteration


class BlockItemsIterPrefetch(object):
    """
        Iterator for the key,val pairs of the block
    """

    def __iter__(self):
        """
        Needed to be considered as an iterable
        """
        return self

    def __init__(self, default_dict):
        """
        Initializes the iterator, and its prefetcher
        Args:
            default_dict: Block to iterate over
        """
        self.pos = 0
        self.keys = []
        self.num_keys = 0
        self.end = False

        self._persistentDict = default_dict
        self._persistentDict.prefetchManager.pipeq_write[0].send(['query'])

    def next(self):
        """
        Returns the values, one by one, contained in the token ranges of the block
        Returns:
            val: .
        """
        self._persistentDict.reads += 1  # STATISTICS
        if self.pos == self.num_keys:
            if self.end:
                raise StopIteration
            else:
                self._persistentDict.prefetchManager.pipeq_write[0].send(['continue'])
                usedpipe = self._persistentDict.prefetchManager.piper_read[0]
                self._persistentDict.cache_prefetchs += 1  # STATISTICS
                self._persistentDict.cache_hits_graph += 'P'  # STATISTICS
                start = time.time()  # STATISTICS
                results = usedpipe.recv()
                self._persistentDict.pending_requests_time += time.time() - start  # STATISTICS
                if len(results) == 0:
                    raise StopIteration
                else:
                    self.keys = []
                    for entry in results:
                        self._persistentDict.cache_hits += 1  # STATISTICS
                        self._persistentDict.cache_hits_graph += 'X'  # STATISTICS
                        start = time.time()  # STATISTICS
                        self._persistentDict.dictCache[entry[0]] = [entry[1], 'Sync']
                        self._persistentDict.cache_hits_time += time.time() - start  # STATISTICS
                        self.keys.append(entry[0])
                    self.num_keys = len(self.keys)
                    self.pos = 0
                    if len(results) < config.batch_size:
                        self.end = True

        if len(self.keys) == 0:
            self._persistentDict.miss += 1  # STATISTICS
            self._persistentDict.cache_hits_graph += '_'  # STATISTICS
            print "Error obtaining block_keys in iter.py"

        value = self._persistentDict.dictCache[self.keys[self.pos]]
        key = self.keys[self.pos]
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

    def __init__(self, block, partition_key, keyspace, table, block_tokens):
        """
        Initializes the iterator
        Args:
            partition_key (String):
            keyspace (String):
            table (String):
            block_tokens (String):
        """
        self._token_pos = 0
        self.block = block
        # TODO this does not work if the primary key is composed
        self._query = config.session.prepare(
            "SELECT * FROM " + keyspace + "." + table + " WHERE " +
            "token(" + ",".join(partition_key) + ") >= ? AND " +
            "token(" + ",".join(partition_key) + ") < ?")
        metadata = config.cluster.metadata
        ringtokens = metadata.token_map
        ran = block_tokens
        ran.sort()
        print "ran:", ran
        last = ringtokens.ring[len(ringtokens.ring) - 1]
        self._token_ranges = []
        max_token = -9223372036854775808
        min_token = 9223372036854775807
        for t in ringtokens.ring:
            if t.value > max_token:
                max_token = t.value
            if t.value < min_token:
                min_token = t.value

        '''
        for t in ringtokens.ring:
            if t.value in ran:
                if t.value == min_token:
                    self._token_ranges.append((-9223372036854775808, min_token))
                    self._token_ranges.append((max_token, 9223372036854775807))
                else:
                    self._token_ranges.append((last.value, t.value))
            last = t
        '''
        print "ringtokens.ring:", ringtokens.ring
        for ind, t in enumerate(ran):
            print "t:", t
            if ind == 0:
                if len(ran) > 1:
                    tok_dist = ran[ind+1] - ran[ind]
                else:
                    for ind2, r in enumerate(ringtokens.ring):
                        if r.value == t:
                            if ind2 < len(ringtokens.ring) - 1:
                                tok_dist = ringtokens.ring[ind2+1].value - ran[ind]
                            else:
                                tok_dist = 9223372036854775807 - ran[ind]
            if ind < len(ran) -1:
                if t == min_token:
                    self._token_ranges.append((9223372036854775807,-9223372036854775808))
                self._token_ranges.append((t, ran[ind+1]))
            else:
                self._token_ranges.append((t, t + tok_dist))
        self._current_iterator = None

    def next(self):
        """
        Returns the values, one by one, contained in the token ranges of the block
        Returns:
            val: .
        """
        if self._current_iterator is not None:
            try:
                to_return = self._current_iterator.next()
                values = []
                for v in self.block.values:
                    try:
                        values.append(getattr(to_return, str(v.encode('utf8'))))
                    except Exception:
                        pass
                if len(values) == 1:
                    values = values[0]
                else:
                    values = tuple(values)
                return values
            except StopIteration:
                # If the current iterator is empty, we try the next token range.
                pass

        if self._token_pos < len(self._token_ranges):
            query = self._query.bind(self._token_ranges[self._token_pos])
            print "query:", query
            self._current_iterator = iter(config.session.execute(query))
            self._token_pos += 1
            return self.next()
        else:
            raise StopIteration


class BlockValuesIterPrefetch(object):
    """
        Iterator for the values of the block
    """

    def __iter__(self):
        """
        Needed to be considered as an iterable
        """
        return self

    def __init__(self, default_dict):
        """
        Initializes the iterator, and its prefetcher
        Args:
            default_dict: Block to iterate over
        """
        self.pos = 0
        self.keys = []
        self.num_keys = 0
        self.end = False

        self._persistentDict = default_dict
        self._persistentDict.prefetchManager.pipeq_write[0].send(['query'])

    def next(self):
        """
        Returns the values, one by one, contained in the token ranges of the block
        Returns:
            val: .
        """
        self._persistentDict.reads += 1  # STATISTICS
        if self.pos == self.num_keys:
            if self.end:
                raise StopIteration
            else:
                self._persistentDict.prefetchManager.pipeq_write[0].send(['continue'])
                usedpipe = self._persistentDict.prefetchManager.piper_read[0]
                self._persistentDict.cache_prefetchs += 1  # STATISTICS
                self._persistentDict.cache_hits_graph += 'P'  # STATISTICS
                start = time.time()  # STATISTICS
                results = usedpipe.recv()
                self._persistentDict.pending_requests_time += time.time() - start  # STATISTICS
                if len(results) == 0:
                    raise StopIteration
                else:
                    self.keys = []
                    for entry in results:
                        self._persistentDict.cache_hits += 1  # STATISTICS
                        self._persistentDict.cache_hits_graph += 'X'  # STATISTICS
                        start = time.time()  # STATISTICS
                        self._persistentDict.dictCache[entry[0]] = [entry[1], 'Sync']
                        self._persistentDict.cache_hits_time += time.time() - start  # STATISTICS
                        self.keys.append(entry[0])
                    self.num_keys = len(self.keys)
                    self.pos = 0
                    if len(results) < config.batch_size:
                        self.end = True

        if len(self.keys) == 0:
            self._persistentDict.miss += 1  # STATISTICS
            self._persistentDict.cache_hits_graph += '_'  # STATISTICS

        value = self._persistentDict.dictCache[self.keys[self.pos]]
        self.pos += 1
        return value[0]


class KeyIter(object):
    """
        Iterator for the blocks of the storageobj
    """

    def __init__(self, keyspace, table, storage_class, object_id, primary_keys, columns):
        """
        Initializes the iterator, and saves the information about the token ranges of each block

        Args:
            keyspace (str) : Cassandra keyspace
            table (str): Cassandra table
            storage_class (str): the full class name of the storage object
            object_id (str): the storage object id
            primary_keys (list(str)): a list of primary keys
            columns (list(str)): a list of values
        """
        self._storage_id = object_id
        self.pos = 0
        self.ring = []
        self.n_blocks = config.number_of_blocks
        self._keyspace = keyspace
        self._table = table
        self._storage_class = storage_class
        primary_keys = map(lambda tupla: tupla[0], primary_keys)
        self._primary_keys = primary_keys
        if 'type' in columns and columns['type'] == 'dict':
            columns = map(lambda tupla: tupla[0], columns['primary_keys']) +\
                      map(lambda tupla: tupla[0], columns['columns'])
        else:
            columns = map(lambda tupla: tupla[0], columns)
        self._columns = columns
        metadata = config.cluster.metadata
        token_to_hosts = dict(map(lambda (tkn, host): (tkn.value, host.address),
                                  metadata.token_map.token_to_host_owner.iteritems()))
        self.ring = KeyIter._calculate_block_ranges(self, token_to_hosts, config.number_of_blocks)

    @staticmethod
    def _calculate_block_ranges(self, token_to_host, n_blocks):
        host_to_tokens = defaultdict(list)
        for t, h in token_to_host.iteritems():
            host_to_tokens[h].append((t, h))

        ring = []
        for tokens in host_to_tokens.values():
            ring += tokens
        
        size_query = "SELECT mean_partition_size, partitions_count FROM system.size_estimates WHERE keyspace_name = \'" + str(self._keyspace) + "\' AND table_name = \'" + str(self._table) + "\';"
        print "size_query:", size_query
        table_size_results = config.session.execute(size_query)
        for row in table_size_results:
            mean_part_size = row.mean_partition_size
            partitions_count = row.partitions_count
        print "##################################################"
        print "Cassandra table:     " + str(self._keyspace) + "." + str(self._table)
        print "mean_partition_size:", mean_part_size
        print "partitions_count:   ", partitions_count
        print "##################################################"
        if mean_part_size == 0:
            ranges_per_token = 1
        else:
            ranges_per_token = int(mean_part_size / 300)
        print "ranges_per_token:", ranges_per_token
        tks = []
        n_tokens = len(token_to_host)
        if n_tokens % n_blocks == 0:
            token_per_block = n_tokens / n_blocks
            tks = [[] for _ in range(n_blocks)]
            for i in range(n_tokens):
                tks[i / token_per_block].append(ring[i])
        elif n_blocks > n_tokens:
            tokens = sorted(ring, key=lambda ring: ring[0])
            tks = [[] for _ in range(len(ring))]
            for i in range(len(tokens)):
                if i == 0:
                    tokdist = (tokens[1][0]-tokens[0][0])/ranges_per_token
                for j in range(0,ranges_per_token):
                    tok = tokens[i][0] + (j * tokdist)
                    tks[i].append((tok,ring[i][1]))
        return tks

    def __iter__(self):
        return self

    def next(self):
        """
        Returns the blocks, one by one, created from the data in the storageobj
        Returns:
            block (Block): a block representing the partition of the dictionary.
        """
        if len(self.ring) == self.pos:
            raise StopIteration

        current_pos = self.ring[self.pos]  # [1]
        host = current_pos[0][1]
        tks = map(lambda a: a[0], current_pos)
        import uuid
        myuuid = str(uuid.uuid1())

        config.session.execute(
            'INSERT INTO ' +
            'hecuba.blocks (blockid,class_name,storageobj_classname,tkns, ksp, tab, obj_type, entry_point, key_list, '
            'value_list, object_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
            [myuuid, "hecuba.iter.Block", self._storage_class, tks, self._keyspace, self._table, 'hecuba', host,
             self._primary_keys, self._columns, self._storage_id])
        b = Block(myuuid, host, self._table, self._keyspace, tks, self._storage_class, self._primary_keys,
                  self._columns, self._storage_id)
        self.pos += 1
        return b
