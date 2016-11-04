# author: G. Alomar
from hecuba.dict import *
from conf.hecuba_params import execution_name, ranges_per_block
from collections import defaultdict
from struct import *
import time
from hecuba import *
import uuid

class Block(object):

    @staticmethod
    def build_remotely(blockid, classname, tkns, entryPoint, port, ksp, tab, dict_name, obj_type):
        return Block(blockid, entryPoint, tab, dict_name, ksp, tkns)




    def __init__(self,blockid, peer, keynames, tablename, blockkeyspace, tokens):
        ''''
        Creates a new block.
        :type blockid: string an unique block identifier
        :type peer: string hostname
        :type keynames: string the Cassandra partition key
        :type tablename: string the name of the collection/table
        :type blockkeyspace: string name of the Cassandra keyspace.
        :type tokens: list of tokens
        '''
        print "Block __init__ ####################################"
        self.blockid = blockid
        self.node = peer
        self.token_ranges = tokens
        self.key_names = keynames
        self.table_name = tablename
        self.keyspace = blockkeyspace
        self.needContext = True
        exec ("from  app.%s import %s"%(tablename.lower(), tablename))
        exec ("self.storageobj = " + str(tablename) + "('" + str(tablename) + "')")
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
        return self.blockid

    def iteritems(self):
        return BlockItemsIter(self)

    def itervalues(self):
        return BlockValuesIter(self)

    def iterkeys(self):
        return BlockIter(self)

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
        print "KeyIter - __init__ ############################################"
        self.pos = 0
        self.ring = []
        self.mypdict = iterable.mypdict
        metadata = cluster.metadata
        ringtokens = metadata.token_map
        tokentohosts = ringtokens.token_to_host_owner
        res = defaultdict(list)

        for tkn, hst in tokentohosts.iteritems():
            res[hst].append(long(((str(tkn).split(':')[1]).replace(' ', '')).replace('>', '')))
            if len(res[hst]) == ranges_per_block:
                self.ring.append((hst, res[hst]))
                res[hst] = []

        self.iterable = iterable
        self.num_peers = len(self.ring)
        self.createIfNot()


    def createIfNot(self):
        try:
            session.execute(
                'CREATE TABLE IF NOT EXISTS hecuba.blocks (blockid text, classname text, tkns list<bigint>, '+
                'entryPoint text , port int, ksp text , tab text , dict_name text , obj_type text, PRIMARY KEY(blockid))')
        except Exception as e:
            print "Error:", e

    def next(self):
        print "KeyIter - next ################################################"
        start = self.pos
        if start == self.num_peers:
            raise StopIteration

        currentRingPos =self.ring[self.pos]    # [1]
        tokens = currentRingPos[1]
        import uuid

        myuuid = str(uuid.uuid1())
        try:
            session.execute('INSERT INTO hecuba.blocks (blockid, classname,tkns, ksp, tab, dict_name, obj_type)'+
                            ' VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                            [myuuid, "hecuba.Block", tokens, self.blockkeyspace, self.mypdict.dict_keynames, self.mypdict.mypo.name, 'hecuba'])
        except Exception as e:
            print "Error:", e
        currringpos = self.ring[self.pos]
        b = Block(myuuid,currringpos[0], self.mypdict.dict_keynames, self.mypdict.mypo.name, self.blockkeyspace,currringpos[1])
        self.pos += 1
        return b

