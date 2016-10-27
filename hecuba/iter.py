# author: G. Alomar
from hecuba.datastore import *
from cassandra.cluster import Cluster
from hecuba.dict import *
from app.qbeastiface import *
from conf.hecuba_params import execution_name, ranges_per_block
import time


class Block(object):
    def __init__(self, peer, keynames, tablename, blockkeyspace):
        self.node = peer[0]
        self.token_ranges = ''
        for ind, position in enumerate(peer):
            if ind == 0:
                self.node = position
            else:
                self.token_ranges = position
        self.key_names = keynames
        self.table_name = tablename
        self.keyspace = blockkeyspace
        self.storageobj = ""
        self.cntxt = ""

    def __iter__(self):
        return BlockIter(self)

    def __getitem__(self, key):
        keys = self.storageobj.keyList[self.storageobj.__class__.__name__]
        exec ("persistentdict = self.storageobj." + str(keys[0]))
        return persistentdict[key]

    def __setitem__(self, key, val):
        keys = self.storageobj.keyList[self.storageobj.__class__.__name__]
        exec ("persistentdict = self.storageobj." + str(keys[0]))
        persistentdict[key] = val

    def getID(self):
        self.key_names = str(self.key_names).replace('\'', '')
        self.key_names = str(self.key_names).replace('(', '')
        self.key_names = str(self.key_names).replace(')', '')
        self.key_names = str(self.key_names).replace(' ', '')
        identifier = "%s_%s_%s_%s" % (self.keyspace, self.key_names, self.table_name, self.token_ranges)
        identifier = identifier.replace(' ', '')
        return identifier

    def iteritems(self):
        return BlockItemsIter(self)

    def itervalues(self):
        return BlockValuesIter(self)

    def iterkeys(self):
        return BlockIter(self)

class IxBlock(Block):
    def __init__(self, peer, keynames, tablename, blockkeyspace, queryLocations):
       super(IxBlock, self).__init__(peer, keynames, tablename, blockkeyspace)
       self.queryLocations = queryLocations
       print "sorprendentemente tambien hemos llegado hasta aqui"
    
class BlockIter(object):
    def __init__(self, iterable):
        self.pos = 0
        self.keys = []
        self.num_keys = 0
        self.iterable = iterable
        self.end = False

        keys = self.iterable.storageobj.keyList[self.iterable.storageobj.__class__.__name__]
        exec ("self.iterable.storageobj." + str(keys[0]) + ".prefetchManager.pipeq_write[0].send(['query'])")

    def next(self):
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
    def __iter__(self):
        return self

    def __init__(self, iterable):
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
    def __iter__(self):
        return self

    def __init__(self, iterable):
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
        self.persistentDict.reads += 1                                                        #STATISTICS
        if self.pos == self.num_keys:
            if self.end:
                raise StopIteration
            else:
                self.persistentDict.prefetchManager.pipeq_write[0].send(['continue'])
                usedpipe = self.persistentDict.prefetchManager.piper_read[0]
                self.persistentDict.cache_prefetchs += 1                                      #STATISTICS
                self.persistentDict.cache_hits_graph += 'P'                                   #STATISTICS
                start = time.time()                                                           #STATISTICS
                results = usedpipe.recv()
                self.persistentDict.pending_requests_time += time.time() - start              #STATISTICS
                if len(results) == 0:
                    raise StopIteration
                else:
                    self.keys = []
                    for entry in results:
                        self.persistentDict.cache_hits += 1                                   #STATISTICS
                        self.persistentDict.cache_hits_graph += 'X'                           #STATISTICS
                        start = time.time()                                                   #STATISTICS
                        self.persistentDict.dictCache[entry[0]] = [entry[1], 'Sync']
                        self.persistentDict.cache_hits_time += time.time() - start            #STATISTICS
                        self.keys.append(entry[0])
                    self.num_keys = len(self.keys)
                    self.pos = 0
                    if len(results) < 100:
                        self.end = True

        if len(self.keys) == 0:
            self.persistentDict.miss += 1                                                     #STATISTICS
            self.persistentDict.cache_hits_graph += '_'                                       #STATISTICS
            print "Error obtaining block_keys in iter.py"

        value = self.persistentDict.dictCache[self.keys[self.pos]]
        self.pos += 1
        return value[0]


class KeyIter(object):
    blockKeySpace = ''

    def __init__(self, iterable):
        self.pos = 0
        self.ring = []
        self.mypdict = iterable.mypdict
        cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)
        session = cluster.connect()
        metadata = cluster.metadata
        ringtokens = metadata.token_map
        tokentohosts = ringtokens.token_to_host_owner
        token_ranges = ''
        starttok = 0
        for i, token in enumerate(ringtokens.ring):
            if ranges_per_block == 1:
                if i == 0:
                    starttok = token
                else:
                    if i < (len(ringtokens.ring)):
                        endtok = token
                        host = str(tokentohosts[starttok])
                        self.ring.append((host, str(i - 1)))
                        starttok = endtok
                    if i == (len(ringtokens.ring) - 1):
                        host = str(tokentohosts[starttok])
                        self.ring.append((host, str(i)))
            else:
                if not (i + 1) % ranges_per_block == 0:
                    if i == 0:
                        starttok = token
                    if (i + 1) % ranges_per_block == 1:
                        starttok = token
                        token_ranges += str(i)
                    else:
                        if i < (len(ringtokens.ring)):
                            endtok = token
                            token_ranges = token_ranges + '_' + str(i)
                            starttok = endtok
                        if i == (len(ringtokens.ring) - 1):
                            token_ranges = token_ranges + '_' + str(i)
                else:
                    if i == 0:
                        starttok = token
                    else:
                        if i < (len(ringtokens.ring)):
                            endtok = token
                            host = str(tokentohosts[starttok])
                            token_ranges = token_ranges + '_' + str(i)
                            self.ring.append((host, token_ranges))
                            token_ranges = ''
                            starttok = endtok

        self.num_peers = len(self.ring)

        session.shutdown()
        cluster.shutdown()

    def next(self):
        start = self.pos
        if start == self.num_peers:
            raise StopIteration
        b = Block(self.ring[self.pos], self.mypdict.dict_keynames, self.mypdict.mypo.name, self.blockkeyspace)
        self.pos += 1
        return b


class IxKeyIter(KeyIter):

    blockKeySpace = ''

    def __init__(self, iterable):
        print "iterable.indexArguments:", iterable.indexarguments
        #super(IxKeyIter, self).__init__(iterable.IxPKeyList)
        super(IxKeyIter, self).__init__(iterable)
        print "InitQuery"
        selects = 'partind'
        keyspace = 'qbeast'
        table = 'MyObj'
        area = [(0,0,0),(10,10,10)]
        precision = 90
        maxResults = 5
        tokens = [0,1,2,3,4,5,6,7,8,9,10]
        qbeastInterface= QbeastIface() # this will be moved to __init__
        qbeastInterface.initQuery(selects, keyspace, table, area, precision, maxResults, tokens)
        
        self.queryLoc = 'queryLocation'

    def next(self):
        start = self.pos
        if start == self.num_peers:
            raise StopIteration
        b = IxBlock(self.ring[self.pos], self.mypdict.dict_keynames, self.mypdict.mypo.name, self.blockkeyspace, self.queryLoc)
        self.pos += 1
        return b

