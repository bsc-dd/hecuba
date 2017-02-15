# author: G. Alomar

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
            print "Not Yet Implemented"
        else:
            partition_key = map(lambda(x,y):(x), self.storageobj._get_default_dict()._primary_keys)
            return BlockIter(self, partition_key, self.keyspace, self.table_name, self.token_ranges, 1)

    def iterkeys(self):
        """
        Obtains the iterator for the keys of the block
        Returns:
            iterkeys(self): list of keys
        """
        if config.prefetch_activated:
            print "Not Yet Implemented"
        else:
            partition_key = map(lambda(x,y):(x), self.storageobj._get_default_dict()._primary_keys)
            return BlockIter(self, partition_key, self.keyspace, self.table_name, self.token_ranges, 1)

    def iteritems(self):
        """
        Obtains the iterator for the key,val pairs of the block
        Returns:
            BlockItemsIter(self): list of key,val pairs
        """
        if config.prefetch_activated:
            print "Not Yet Implemented"
        else:
            partition_key = map(lambda(x,y):(x), self.storageobj._get_default_dict()._primary_keys)
            return BlockIter(self, partition_key, self.keyspace, self.table_name, self.token_ranges, 2)

    def itervalues(self):
        """
        Obtains the iterator for the values of the block
        Returns:
            BlockValuesIter(self): list of values
        """
        if config.prefetch_activated:
            print "Not Yet Implemented"
        else:
            partition_key = map(lambda(x,y):(x), self.storageobj._get_default_dict()._primary_keys)
            return BlockIter(self, partition_key, self.keyspace, self.table_name, self.token_ranges, 3)

class BlockIter(object):
    """
        Iterator for the values of the block
    """

    def __iter__(self):
        """
        Needed to be considered as an iterable
        """
        return self

    def __init__(self, block, partition_key, keyspace, table, block_tokens, mode):
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
        self.mode = mode
        self._query = config.session.prepare(
            "SELECT * FROM " + keyspace + "." + table + " WHERE " +
            "token(" + ",".join(partition_key) + ") >= ? AND " +
            "token(" + ",".join(partition_key) + ") < ?")
        ran = block_tokens
        self._token_ranges = []
        min_token = -9223372036854775808

        for ind, t in enumerate(ran):
            if t[0] == min_token:
                self._token_ranges.append((9223372036854775807, -9223372036854775808))
            self._token_ranges.append(t)
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
                if self.mode < 3:
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
                if self.mode > 1:
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
                if self.mode == 1:
                    return keys
                if self.mode == 2:
                    return keys, values
                if self.mode == 3:
                    return values
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

        self._query = config.session.prepare(
            'INSERT INTO ' +
            'hecuba.blocks (blockid,class_name,storageobj_classname, tkns, ksp, tab, obj_type, entry_point, key_list, '
            'value_list, object_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)')

    @staticmethod
    def _calculate_block_ranges(self, token_to_host, n_blocks):
        host_to_tokens = defaultdict(list)
        for t, h in token_to_host.iteritems():
            host_to_tokens[h].append((t, h))

        ring = []
        for tokens in host_to_tokens.values():
            ring += tokens

        n_tokens = len(token_to_host)
        tokens = sorted(ring, key=lambda ring: ring[0])
        ranges_per_block = config.ranges_per_block
        self.blocks_to_ips = []
        print "tokens:     ", tokens
        print "len(tokens):", len(tokens)
        print "n_blocks:        ", n_blocks
        print "ranges_per_token:", ranges_per_block
        total_tokens = ranges_per_block * n_blocks
        print "total_tokens:    ", total_tokens
        if total_tokens > len(tokens):
            ranges_per_token = total_tokens / len(tokens)
            tks = defaultdict(list)
            for i in range(len(tokens)):
                for j in range(1, ranges_per_token + 1):
                    if j == 1:
                        if i < len(tokens) - 1:
                            tok_dist = (tokens[i + 1][0] - tokens[i][0]) / ranges_per_token
                        else:
                            tok_dist = (9223372036854775807 - tokens[i][0]) / ranges_per_token
                        first_tok = tokens[i][0]
                    last_tok = first_tok + tok_dist
                    if last_tok > 9223372036854775807:
                        last_tok = 9223372036854775807
                    tks[str(i), str(ring[i][1])].append((int(first_tok), int(last_tok)))
                    if (str(i), str(ring[i][1])) not in self.blocks_to_ips:
                        self.blocks_to_ips.append((str(i), str(ring[i][1])))
                    first_tok = last_tok + 1
        else:
            tks = defaultdict(list)
            merge_quantity = (len(tokens) / total_tokens) + 1
            print "merge_quantity:", merge_quantity
            for i in range(len(tokens)):
                if i < len(tokens) - 1:
                    print "ring[i][1]:", ring[i][1]
                    if i % merge_quantity == 0:
                        first_tok = tokens[i][0]
                    elif i % merge_quantity == merge_quantity - 1:
                        last_tok = tokens[i+1][0] - 1
                        if last_tok > 9223372036854775807:
                            last_tok = 9223372036854775807
                        print str(first_tok) + ", " + str(last_tok) + ", " + str(ring[i][1])
                        tks[str(i), str(ring[i][1])].append((int(first_tok), int(last_tok)))
                        if (str(i), str(ring[i][1])) not in self.blocks_to_ips:
                            self.blocks_to_ips.append((str(i), str(ring[i][1])))
                else:
                    if i % merge_quantity == 0:
                        first_tok = tokens[i][0]
                    elif i % merge_quantity == merge_quantity - 1:
                        last_tok = 9223372036854775807
                        print str(first_tok) + ", " + str(last_tok) + ", " + str(ring[i][1])
                        tks[str(i), str(ring[i][1])].append((int(first_tok), int(last_tok)))
                        if (str(i), str(ring[i][1])) not in self.blocks_to_ips:
                            self.blocks_to_ips.append((str(i), str(ring[i][1])))



        '''
        if n_blocks >= n_tokens:
            tks = [[] for _ in range(len(ring))]
            for i in range(len(tokens)):
                for j in range(1, ranges_per_token + 1):
                    if j == 1:
                        if i < len(tokens) - 1:
                            tok_dist = (tokens[i + 1][0] - tokens[i][0]) / ranges_per_block
                        else:
                            tok_dist = (9223372036854775807 - tokens[i][0]) / ranges_per_block
                        first_tok = tokens[i][0]
                    last_tok = first_tok + tok_dist
                    if last_tok > 9223372036854775807:
                        last_tok = 9223372036854775807
                    tks[i].append(((int(first_tok), int(last_tok)), ring[i][1]))
                    first_tok = last_tok
        else:
            merge_quantity = 2
            ranges_per_block = len(tokens) / 2
            print "merge_quantity:  ", merge_quantity
            print "len(tokens):     ", len(tokens)
            print "ranges_per_block:", ranges_per_block
            print "ring:            ", ring
            tks = [[] for _ in range(ranges_per_block)]
            inserted = 0
            for i in range(len(tokens)):
                if i < len(tokens) - 1:
                    print "ring[i][1]:", ring[i][1]
                    if i % merge_quantity == 0:
                        first_tok = tokens[i][0]
                    elif i % merge_quantity == merge_quantity - 1:
                        last_tok = tokens[i+1][0] - 1
                        if last_tok > 9223372036854775807:
                            last_tok = 9223372036854775807
                        print str(first_tok) + ", " + str(last_tok) + ", " + str(ring[i][1])
                        tks[inserted].append(((int(first_tok), int(last_tok)), ring[i][1]))
                        inserted += 1
                else:
                    if i % merge_quantity == 0:
                        first_tok = tokens[i][0]
                    elif i % merge_quantity == merge_quantity - 1:
                        last_tok = 9223372036854775807
                        print str(first_tok) + ", " + str(last_tok) + ", " + str(ring[i][1])
                        tks[inserted].append(((int(first_tok), int(last_tok)), ring[i][1]))
                        inserted += 1
        '''
        print "tks:", tks
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

        curr_key = self.blocks_to_ips[self.pos]
        current_pos = self.ring[curr_key]  # [1]
        print "current_pos:", current_pos
        host = curr_key[1]
        tks = map(lambda a: (a[0], a[1]), current_pos)
        import uuid
        myuuid = str(uuid.uuid1())

        query = self._query.bind([myuuid, "hecuba.iter.Block", self._storage_class, tks, self._keyspace, self._table,
                                  'hecuba', host, self._primary_keys, self._columns, self._storage_id])
        config.session.execute(query)

        b = Block(myuuid, host, self._table, self._keyspace, tks, self._storage_class, self._primary_keys,
                  self._columns, self._storage_id)
        self.pos += 1
        return b
