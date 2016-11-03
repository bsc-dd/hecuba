# author: G. Alomar
from hecuba.datastore import *
from cassandra.cluster import Cluster
from hecuba.dict import *
from hecuba.qbeastiface import *
from qbeastIntegration.ttypes import Result
from conf.hecuba_params import execution_name, ranges_per_block
from collections import defaultdict
from struct import *
import time
import uuid

class Block(object):
    def __init__(self, peer, keynames, tablename, blockkeyspace):
        print "IxBlock __init__ ####################################"
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
    def __init__(self, peer, keynames, tablename, blockkeyspace, myuuid):
        print "IxBlock __init__ ####################################"
        self.node = peer
        self.key_names = keynames
        self.table_name = tablename
        self.keyspace = blockkeyspace
        self.storageobj = ""
        self.cntxt = ""
        self.myuuid = myuuid

    def iteritems(self):
        print "in IxBlock.iteritems()"
        return IxBlockItemsIter(self)
   
    def getID(self):
        print "IxBlock getID #######################################"
        return self.myuuid 

    def itervalues(self): #to implement
        print "IxBlock itervalues ##################################"
        pass
        #return BlockValuesIter(self)


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


class IxBlockItemsIter(object):
    def __iter__(self):
        print "IxBlockItemsIter.__iter__"
        return self

    def __init__(self, iterable):
        print "IxBlockItemsIter.__init__"
        self.toReturn = []
        '''
        Attributes:
         - hasMore
         - count
         - metadata
         - data
        '''
        txtToPack = 'OriginalText'
        self.result = (False,
                   6,
                   {0: "BIGINT", 1: "BLOB", 2: "BOOLEAN", 3: "DOUBLE", 4: "FLOAT", 5: "INET", 6: "INT", 7: "LIST", 8: "MAP", 9: "SET", 10: "TEXT", 11: "TIMESTAMP", 12: "TIMEUUID", 13: "UUID"},
                   [(0,pack("<l",1234567890)),(2,pack("<?",True)),(3,pack("<d",234.567)),(4,pack("<f",345.6)),(6,pack("<b",15)),(10,pack("I%ds" % len(txtToPack),txtToPack))])
        self.equivs = self.result[2]
        self.toReturn = self.result[3]
        print "self.toReturn:", self.toReturn

    def next(self):
        # do gets from Qbeast until done
        # for every get, iterate over results and save them in a list in the IxBlockItemsIter object
        # while the object has values in the list, pop them one by one and call next again
        print "IxBlockItemsIter.next"
        if self.result[0] == False and len(self.toReturn) == 0:
            raise StopIteration
        toRet = self.toReturn.pop()
        print "toRet: ", toRet
        if toRet[0] == 0:
            return (0,unpack("<l",toRet[1]))
        if toRet[0] == 2:
            return (1,unpack("<?",toRet[1]))
        if toRet[0] == 3:
            return (3,unpack("<d",toRet[1]))
        if toRet[0] == 4:
            return (4,unpack("<f",toRet[1]))
        if toRet[0] == 6:
            return (6,unpack("<b",toRet[1]))
        if toRet[0] == 10:
            size=len(toRet[1])
            print "size:", size
            return (10,unpack("I%ds"%(size),toRet[1]))[1]


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
        cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)
        session = cluster.connect()
        metadata = cluster.metadata
        ringtokens = metadata.token_map
        tokentohosts = ringtokens.token_to_host_owner
        token_ranges = ''
        starttok = 0
        self.tokenList = []
        for i, token in enumerate(ringtokens.ring):
            if ranges_per_block == 1:
                if i == 0:
                    starttok = token
                else:
                    if i < (len(ringtokens.ring)):
                        endtok = token
                        host = str(tokentohosts[starttok])
                        self.ring.append((host, str(i - 1)))
                        self.tokenList.append(int(token.value))
                        starttok = endtok
                    if i == (len(ringtokens.ring) - 1):
                        host = str(tokentohosts[starttok])
                        self.ring.append((host, str(i)))
                        self.tokenList.append(int(token.value))
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
                            self.tokenList.append(int(token.value))
                            token_ranges = ''
                            starttok = endtok

        self.num_peers = len(self.ring)
        session.shutdown()
        cluster.shutdown()

    def next(self):
        print "KeyIter - next ################################################"
        start = self.pos
        if start == self.num_peers:
            raise StopIteration

        '''
        currentRingPos =self.ring[self.pos]    # [1]
        tokens = currentRingPos[1]

        cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)
        session = cluster.connect()
        try:
            session.execute('CREATE TABLE IF NOT EXISTS hecuba.blocks (blockid text, tkns list<bigint>, entryPoint text , port int, ksp text , tab text , dict_name text , obj_type text, PRIMARY KEY(blockid))')
        except Exception as e:
            print "Error:", e
        myuuid = str(uuid.uuid1())
        try:
            session.execute('INSERT INTO hecuba.blocks (blockid, tkns, ksp, tab, dict_name, obj_type, entrypoint, port) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)',
                                                       [myuuid,  tokens, self.blockkeyspace, self.mypdict.dict_keynames, self.mypdict.mypo.name,'hecuba','localhost',1] )
        except Exception as e:
            print "Error:", e
        session.shutdown()
        cluster.shutdown()
        '''

        b = Block(self.ring[self.pos], self.mypdict.dict_keynames, self.mypdict.mypo.name, self.blockkeyspace)
        self.pos += 1
        return b


class IxKeyIter(KeyIter):

    blockKeySpace = ''

    def __init__(self, iterable):
        print "KeyIter - __init__ ############################################"
        self.pos = 0
        self.ring = []
        self.mypdict = iterable.mypdict
        cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)
        session = cluster.connect()
        metadata = cluster.metadata
        ringtokens = metadata.token_map
        tokentohosts = ringtokens.token_to_host_owner
        res = defaultdict(list)

        for tkn, hst in tokentohosts.iteritems():
            res[hst].append(long(((str(tkn).split(':')[1]).replace(' ','')).replace('>','')))
            if len(res[hst]) == ranges_per_block:
                self.ring.append((hst, res[hst]))
                res[hst] = []

        self.num_peers = len(self.ring)
        session.shutdown()
        cluster.shutdown()
        self.iterable = iterable
        
    def next(self):
        print "IxKeyIter - next ##############################################"
        start = self.pos
        if start == self.num_peers:
            raise StopIteration

        minarguments = {}
        maxarguments = {}
        for argument in self.iterable.indexarguments:
            if '<' in str(argument):
                splitarg = (str(argument).replace(' ','')).split('<')
                val = str(splitarg[0])
                maxarguments[val] = int(splitarg[1])
            if '>' in str(argument):
                splitarg = (str(argument).replace(' ','')).split('>')
                val = str(splitarg[0])
                minarguments[val] = int(splitarg[1])
        selects = 'partind' # shouldnt be hardcoded
        keyspace = 'qbeast' # shouldnt be hardcoded
        table = self.__class__.__name__
        area = [(minarguments['x'],minarguments['y'],minarguments['z']),(maxarguments['x'],maxarguments['y'],maxarguments['z'])]  #[(0,0,0),(10,10,10)]
        precision = 90
        maxResults = 5
        print "self.pos:", self.pos
        currentRingPos =self.ring[self.pos]    # [1]
        tokens = currentRingPos[1]
        qbeastInterface= QbeastIface() # this will be moved to __init__
        print "InitQuery"
        self.queryLoc = qbeastInterface.initQuery(selects, keyspace, table, area, precision, maxResults, tokens)

        cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)
        session = cluster.connect()
        try:
            session.execute('CREATE TABLE IF NOT EXISTS hecuba.blocks (blockid text, tkns list<bigint>, entryPoint text , port int, ksp text , tab text , dict_name text , obj_type text, PRIMARY KEY(blockid))')
        except Exception as e:
            print "Error:", e
        '''
        try:
            session.execute('TRUNCATE hecuba.blocks')
        except Exception as e:
            print "Error:", e
        '''
        myuuid = str(uuid.uuid1())
        try:
            session.execute('INSERT INTO hecuba.blocks (blockid, tkns, ksp, tab, dict_name, obj_type, entrypoint, port) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)',
                                                       [myuuid,  tokens, self.blockkeyspace, self.mypdict.dict_keynames, self.mypdict.mypo.name,'qbeast','localhost',1] )
        except Exception as e:
            print "Error:", e
        session.shutdown()
        cluster.shutdown()

        b = IxBlock(currentRingPos, self.mypdict.dict_keynames, self.mypdict.mypo.name, self.blockkeyspace, myuuid)
        self.pos += 1
        return b


