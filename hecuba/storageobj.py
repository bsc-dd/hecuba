# author: G. Alomar
from cassandra.cluster import Cluster
from collections import defaultdict
from hecuba.Plist import PersistentKeyList
from hecuba.dict import PersistentDict
from hecuba.datastore import *
from conf.hecuba_params import execution_name
from conf.apppath import apppath
import time
import glob
import os

class StorageObj(object):

    keyList = defaultdict(list)
    nextKeys = []
    cntxt = ''

    def __init__(self, name=None):
        print "storageobj __init__ ####################################"
        print self.__class__.__name__
        setattr(self, 'name', None)
        setattr(self, 'persistent', False)
        setattr(self, 'indexed', False)
        self.getByName(name)

    def init_prefetch(self, block):
        keys = self.keyList[self.__class__.__name__]
        exec("self." + str(keys[0]) + ".init_prefetch(block)")

    def end_prefetch(self):
        keys = self.keyList[self.__class__.__name__]
        exec("self." + str(keys[0]) + ".end_prefetch()")

    def getByName(self, name):
        print "storageobj getByName ####################################"
        print self.__class__.__name__
        if name is None:
            self.persistent = False
        else:
            self.persistent = True
            self.name = name

        keyspace = 'config' + execution_name
        cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)
        session = ''
        try:
            session = cluster.connect()
        except Exception as e:
            print "connection could not be set", e

        try:
            session.set_keyspace(keyspace)
        except Exception as e:
            print "keyspace could not be set", e

        execution = []

        dictname = "\"" + self.__class__.__name__ + "\""

        query = "SELECT * FROM " + keyspace + "." + dictname + ";"
        numtables = 0
        try:
            execution = session.execute(query)
        except Exception as e:
            print "Error: Cannot retrieve data from table", dictname, ". Exception: ", e

        currtable = {}
        for ind, row in enumerate(execution):
            currtable[ind] = row
            if row[0] > numtables:
                numtables = str(row[0])

        initial = numtables
        for x in range(0, int(numtables)):
            yeskeys = "("
            notkeys = ""
            for key, row in currtable.iteritems():
                if int(row[0]) == int(initial):
                    if row[3] == "yes":
                        yeskeys = yeskeys + "\'" + row[1] + "\', "
                    else:
                        notkeys = notkeys + "\'" + row[1] + "\', "
            if int(initial) > 0:
                initial = int(initial) - 1
            else:
                initial = numtables
            notk = notkeys.split(',')[0]
            notk = str(notk)[1:len(notk)-1]
            notkeys = notkeys[0:len(notkeys) - 2]
            yeskeys = yeskeys[0:len(yeskeys) - 2]
            yeskeys += ")"
            exec("self." + str(notk) + " = PersistentDict(self, (" + notkeys + "), " + yeskeys + ")")

        filestoparse = glob.glob(apppath + "/app/*.py")
        for ftp in filestoparse:
            f = open(ftp, 'r')
            incomment = False
            classf = False
            for line in f:
                if "class " in line:
                    if ("(StorageObj)" in line) or ("(StorageObjIx)" in line):
                        classf = True
                        if not incomment:
                            completeline = str(line)
                            line = line.split(" ")
                            line = line[1]
                            line = line.split("(")
                            classn = line[0]
                            if classn == self.__class__.__name__:
                                if "(StorageObj)" in completeline:
                                    print "self.indexed = False"
                                    self.indexed = False
                                if "(StorageObjIx)" in completeline:
                                    print "self.indexed = True"
                                    self.indexed = True
                if "'''" in line:
                    if classf:
                        if not incomment:
                            incomment = True
                        else:
                            incomment = False

        session.shutdown()
        cluster.shutdown()

    def make_persistent(self, name):
        print "storageobj make_persistent ####################################"
        print self.__class__.__name__

        self.name = name
        cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)

        keyspace = 'config' + execution_name

        session = cluster.connect(keyspace)

        dictname = "\"" + self.__class__.__name__ + "\""

        query = "SELECT * FROM " + keyspace + "." + dictname + ";"
        try:
            execution = session.execute(query)
        except Exception as e:
            print "Error: Cannot retrieve data from table", dictname, ". Exception: ", e

        yeskeys = "("
        yeskeystypes = ""
        notkeystypes = ""
        atribstypes = []
        typesy = []
        typesn = []
        # Row(dictid, dataname, datatype, iskey, keyorder)
        #     row[0]   row[1]    row[2]  row[3]   row[4]
        for row in execution:
            if row[3] == "yes":                                                             # if element is key
                if row[1] not in typesy:
                    yeskeys = yeskeys + row[1] + ", "
                    typesy.append(row[1])
                if row[1] not in typesn:                                                    # if key name + datatype hasn't been processed yet
                    yeskeystypes = yeskeystypes + row[1] + " " + row[2] + ", "              # save name and datatype
                    typesn.append(row[1])
                if row[2] == 'int':                                                         # if key is an int
                    if row[1] not in atribstypes:
                        atribstypes.append(row[1])
                if row[2] == 'text':                                                        # if key is a string
                    if row[1] not in atribstypes:
                        atribstypes.append(row[1])
            else:                                                                           # if element is not key
                if row[1] not in typesn:
                    notkeystypes = notkeystypes + row[1] + " " + row[2] + ", "
                    typesn.append(row[1])

        notkeystypes = notkeystypes[0:len(notkeystypes) - 2]
        yeskeystypes = yeskeystypes[0:len(yeskeystypes) - 2]
        yeskeys = yeskeys[0:len(yeskeys) - 2]
        yeskeys += ")"

        try:
            session.set_keyspace(execution_name)
        except Exception as e:
            print "keyspace could not be set", e

        querytable = "CREATE TABLE " + execution_name + ".\"" + str(name) + "\" (%s, %s, PRIMARY KEY%s);" % (yeskeystypes, notkeystypes, yeskeys)
        try:
            session.execute(querytable)
        except Exception as e:
            print "error in querytable:", querytable
            print "Object", self.name, "cannot be created in persistent storage", e
            #pass

        keys = self.keyList[self.__class__.__name__]
        for key in self.split():
            exec("val = self." + str(keys[0]) + "[" + str(key) + "]")
            query = "INSERT INTO " + str(execution_name) + ".\"" + str(name) + "\"(" + str(yeskeys[1:len(yeskeys)-1]) + ", " + str(keys[0]) + ") VALUES (" + str(key) + ", " + str(val) + ");"
            session.execute(query)

        for key, variable in vars(self).iteritems():
            if str(type(variable)) == "<type 'int'>":
                querytable = "INSERT INTO config" + str(execution_name) + ".attribs(dictname, dataname, datatype, datavalue) VALUES ( \'" + str(self.name) + "\', \'" + str(key) + "\', 'int', \'" + str(variable) + "\');"
                session.execute(querytable)
            if str(type(variable)) == "<type 'str'>":
                if not str(key) == 'name':
                    querytable = "INSERT INTO config" + str(execution_name) + ".attribs(dictname, dataname, datatype, datavalue) VALUES ( \'" + str(self.name) + "\', \'" + str(key) + "\', 'str', \'" + str(variable) + "\');"
                    session.execute(querytable)
            if str(type(variable)) == "<type 'list'>":
                querytable = "CREATE TABLE " + execution_name + ".\"" + str(name) + str(key) + "\" (position int, type text, value text, PRIMARY KEY (position));"
                try:
                    session.execute(querytable)
                except Exception as e:
                    #print "Object", self.name, "cannot be created in persistent storage", e
                    pass
                for ind, value in enumerate(variable):
                    if str(type(value)) == "<type 'int'>":
                        querytable = "INSERT INTO " + execution_name + ".\"" + str(name) + str(key) + "\" (position, type, value) VALUES ( " + str(ind) + ", \'int\', \'" + str(value) + "\');"
                    if str(type(value)) == "<type 'str'>":
                        querytable = "INSERT INTO " + execution_name + ".\"" + str(name) + str(key) + "\" (position, type, value) VALUES ( " + str(ind) + ", \'str\', \'" + str(value) + "\');"
                    try:
                        session.execute(querytable)
                    except Exception as e:
                        print "Object", self.name, "cannot be inserted in persistent storage", e

        self.persistent = True

        session.shutdown()
        cluster.shutdown()

    def saveToDDBB(self, name):

        self.name = name
        cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)

        keyspace = 'config' + execution_name

        session = cluster.connect(keyspace)

        for key, variable in vars(self).iteritems():
            if str(type(variable)) == "<type 'int'>":
                querytable = "INSERT INTO config" + str(execution_name) + ".attribs(dictname, dataname, datatype, datavalue) VALUES ( \'" + str(self.name) + "\', \'" + str(key) + "\', 'int', \'" + str(variable) + "\');"
                session.execute(querytable)
            if str(type(variable)) == "<type 'str'>":
                if not str(key) == 'name':
                    querytable = "INSERT INTO config" + str(execution_name) + ".attribs(dictname, dataname, datatype, datavalue) VALUES ( \'" + str(self.name) + "\', \'" + str(key) + "\', 'str', \'" + str(variable) + "\');"
                    session.execute(querytable)
            if str(type(variable)) == "<type 'list'>":
                query = "TRUNCATE %s.\"%s\";" % (execution_name, self.name + str(key))
                try:
                    session.execute(query)
                except Exception as e:
                    print "Object", self.name, "cannot be emptied in persistent storage saveToDDBB:", e
                querytable = "CREATE TABLE " + execution_name + ".\"" + str(self.name) + str(key) + "\" (position int, type text, value text, PRIMARY KEY (position));"
                try:
                    session.execute(querytable)
                except Exception as e:
                    #print "Object", self.name, "cannot be created in persistent storage", e
                    pass
                for ind, value in enumerate(variable):
                    if str(type(value)) == "<type 'int'>":
                        querytable = "INSERT INTO " + execution_name + ".\"" + str(self.name) + str(key) + "\" (position, type, value) VALUES ( " + str(ind) + ", \'int\', \'" + str(value) + "\');"
                    if str(type(value)) == "<type 'str'>":
                        querytable = "INSERT INTO " + execution_name + ".\"" + str(self.name) + str(key) + "\" (position, type, value) VALUES ( " + str(ind) + ", \'str\', \'" + str(value) + "\');"
                    session.execute(querytable)

        session.shutdown()
        cluster.shutdown()
    
    '''
    def iteritems(self):
        print "storageobj iteritems ####################################"
        print "Data needs to be accessed through a block"
        return [] # self
    '''
    #'''
    # new iteritems
    def iteritems(self):
        print "storageobj iteritems ####################################"
        keys = self.keyList[self.__class__.__name__]
        exec ("self.pKeyList = PersistentKeyList(self." + str(keys[0]) + ")")
        return self # a
    #'''
    def itervalues(self):
        print "Data needs to be accessed through a block"
        return [] # self
    '''
    def iteritems(self):
        keys = self.keyList[self.__class__.__name__]
        exec ("a = PersistentKeyList(self." + str(keys[0]) + ")")
        return a # self
    '''

    def increment(self, target, value):
        self[target] = value

    def empty_persistent(self):
        keys = self.keyList[self.__class__.__name__]
        exec("self." + str(keys[0]) + ".dictCache.cache = {}")

        cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)
        session = cluster.connect(execution_name)

        query = "TRUNCATE %s.\"%s\";" % (execution_name, self.name)

        sessionexecute = 0
        sleeptime = 0.05
        while sessionexecute < 10:
            try:
                session.execute(query)
                sessionexecute = 10
            except Exception as e:
                time.sleep(sleeptime)
                sleeptime *= 2
                if sessionexecute == 9:
                    print "Error: Cannot empty object. Exception: ", e
                    session.shutdown()
                    raise e
                sessionexecute += 1

        session.shutdown()
        cluster.shutdown()

    def delete_persistent(self):
        keys = self.keyList[self.__class__.__name__]
        exec("self." + str(keys[0]) + ".dictCache.cache = {}")

        cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)
        session = cluster.connect(execution_name)
        self.persistent = False

        query = "TRUNCATE %s.\"%s\";" % (execution_name, self.name)
        try:
            session.execute(query)
        except Exception as e:
            print "Object", str(self.name), "cannot be emptied:", str(e)
            return

        query = "DROP COLUMNFAMILY %s.\"%s\";" % (execution_name, self.name)
        try:
            session.execute(query)
        except Exception as e:
            print "Object", str(self.name), "cannot be deleted from persistent storage:", str(e)
            return

        session.shutdown()
        cluster.shutdown()

    def __contains__(self, key):
        keys = self.keyList[self.__class__.__name__]
        if len(keys) > 1:
            raise KeyError
        else:
            exec("a = self." + str(keys[0]) + ".__contains__(key)")
            return a

    def has_key(self, key):
        keys = self.keyList[self.__class__.__name__]
        if len(keys) > 1:
            raise KeyError
        else:
            exec("a = self." + str(keys[0]) + ".has_key(key)")
            return a

    def finalize(self):
        keyspace = 'config' + execution_name
        try:
            cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)
            session = cluster.connect()
            execution = session.execute("SELECT * FROM " + keyspace + ".\"access\";")
            for row in execution:
                if row[1] == 0:
                    try:
                        session.execute("DROP COLUMNFAMILY " + keyspace + ".\"" + row[0] + "Types\";")
                    except Exception as e:
                        print "Error in finalize DROP 1:", e
                    try:
                        session.execute("DELETE FROM " + keyspace + ".\"access\" WHERE classtable = \'" + row[0] + "\';")
                    except Exception as e:
                        print "Error in finalize DELETE:", e
                execution = session.execute("SELECT * FROM " + keyspace + ".\"access\";")
            if len(execution) == 0:
                try:
                    session.execute("DROP COLUMNFAMILY " + keyspace + ".\"access\";")
                except Exception as e:
                    print "Error in finalize DROP 2:", e
            session.shutdown()
            cluster.shutdown()
        except Exception as e:
            print "Error in finalize:", e

    def __getattr__(self,key):
        #print "storageobj __getattr__"
        #print "self:                   ", self
        #print "key:                    ", key
        toreturn = ''
        if not key == 'persistent':
            if hasattr(self, 'persistent'):
                if not super(StorageObj, self).__getattribute__('persistent'):
                    if hasattr(self,key):
                        toreturn = super(StorageObj, self).__getattribute__(key)
                        return toreturn
                    else:
                        raise KeyError

                else:
                    cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)

                    keyspace = 'config' + execution_name

                    session = cluster.connect(keyspace)

                    query = "SELECT * FROM config" + str(execution_name) + ".attribs WHERE dictname = '" + self.__class__.__name__ + "' AND dataname = '" + str(key) + "';"
                    try:
                        result = session.execute(query)
                    except Exception as e:
                        #print "Object", self.name, "cannot be selected in persistent storage __getattr__:", e
                        pass

                    for row in result:
                        restype = str(row[2]).split('_')
                        if restype[0] == 'list':
                            values = []
                            query2 = "SELECT * FROM " + str(row[3]) + ";"
                            try:
                                listvals = session.execute(query2)
                            except Exception as e:
                                print "error obtaining list values:", e
                            for valrow in listvals:
                                if str(valrow[1]) == 'str':
                                    values.append(str(valrow[2]))
                                else:
                                    values.append(int(str(valrow[2]).replace('\'','')))
                            toreturn = values
                        else:
                            toreturn = row[3]

                    session.shutdown()
                    cluster.shutdown()

                    return toreturn
            else:
                return super(StorageObj, self).__getattribute__(key)
        else:
            return super(StorageObj, self).__getattribute__(key)

    def __setattr__(self, key, value):
        #print "storageobj - __setattr__"
        #print "self:                   ", self
        #print "key:                    ", key
        if str(type(value)) == "<class 'hecuba.dict.PersistentDict'>":
            super(StorageObj, self).__setattr__(key, value)
        else:
            if not (str(key) == 'indexed') and not (str(key) == 'name') and not (str(key) == 'persistent') and not (str(key) == 'cntxt'):
                if hasattr(self, 'persistent'):
                    if not self.persistent:
                        super(StorageObj, self).__setattr__(key, value)
                    else:
                        cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)

                        keyspace = 'config' + execution_name

                        session = cluster.connect(keyspace)

                        if type(value) == list:
                            querytable = "CREATE TABLE " + execution_name + ".\"" + str(self.__class__.__name__) + str(key) + "\" (position int, type text, value text, PRIMARY KEY (position));"
                            try:
                                session.execute(querytable)
                            except Exception as e:
                                #print "Object", self.__class__.__name__, "cannot be created in persistent storage", e
                                pass
                            query = "TRUNCATE %s.\"%s\";" % (execution_name, str(self.__class__.__name__) + str(key))
                            try:
                                session.execute(query)
                            except Exception as e:
                                #print "Object", self.name, "cannot be emptied in persistent storage:", e
                                pass
                            strtypeval = ''
                            for ind, val in enumerate(value):
                                if str(type(val)) == "<type 'int'>":
                                    strtypeval = 'int'
                                    querytable = "INSERT INTO " + execution_name + ".\"" + str(self.name) + str(key) + "\" (position, type, value) VALUES ( " + str(ind) + ", \'int\', \'" + str(val) + "\');"
                                if str(type(val)) == "<type 'str'>":
                                    strtypeval = 'str'
                                    querytable = "INSERT INTO " + execution_name + ".\"" + str(self.name) + str(key) + "\" (position, type, value) VALUES ( " + str(ind) + ", \'str\', \'" + str(val) + "\');"
                                try:
                                    session.execute(querytable)
                                except Exception as e:
                                    print "Object", str(value), "cannot be inserted in persistent storage", e
                            if strtypeval == 'str':
                                querytable = "INSERT INTO config" + str(execution_name) + ".attribs(dictName, dataName, dataType, dataValue) VALUES ( \'" + str(self.__class__.__name__) + "\', \'" + str(key) + "\', \'list_str\', \'" + str(execution_name) + ".\"" + str(self.__class__.__name__) + str(key) + "\"\')"
                            else:
                                querytable = "INSERT INTO config" + str(execution_name) + ".attribs(dictName, dataName, dataType, dataValue) VALUES ( \'" + str(self.__class__.__name__) + "\', \'" + str(key) + "\', \'list_int\', \'" + str(execution_name) + ".\"" + str(self.__class__.__name__) + str(key) + "\"\')"
                            try:
                                session.execute(querytable)
                            except Exception as e:
                                print "Object", str(value), "cannot be inserted in persistent storage", e

                        else:
                            if type(value) == int:
                                querytable = "INSERT INTO config" + str(execution_name) + ".attribs(dictname, dataname, datatype, datavalue) VALUES ( \'" + str(self.__class__.__name__) + "\', \'" + str(key) + "\', 'int', \'" + str(value) + "\');"
                                try:
                                    session.execute(querytable)
                                except Exception as e:
                                    print "Object", str(value), "cannot be inserted in persistent storage", e
                            if type(value) == str:
                                if not str(key) == 'name':
                                    querytable = "INSERT INTO config" + str(execution_name) + ".attribs(dictname, dataname, datatype, datavalue) VALUES ( \'" + str(self.__class__.__name__) + "\', \'" + str(key) + "\', 'str', \'" + str(value) + "\');"
                                    try:
                                        session.execute(querytable)
                                    except Exception as e:
                                        print "Object", str(value), "cannot be inserted in persistent storage", e

                        session.shutdown()
                        cluster.shutdown()

                        if hasattr(self, key):
                            try:
                                delattr(self, key)
                            except Exception as e:
                                pass
                else:
                    super(StorageObj, self).__setattr__(key, value)
            else:
                super(StorageObj, self).__setattr__(key, value)

    def getID(self):
        identifier = "%s_%s" % (self.name, '1')
        identifier = identifier.replace(" ", "")
        identifier = identifier.replace("'", "")
        identifier = identifier.replace("(", "")
        identifier = identifier.replace(")", "")
        return identifier

    def split(self):
        keys = self.keyList[self.__class__.__name__]
        if not self.persistent:
            exec("a = dict.keys(self." + str(keys[0]) + ")")
            return a
        else:
            exec("a = PersistentKeyList(self." + str(keys[0]) + ")")
            return a

    def __additem__(self, key, other):
        keys = self.keyList[self.__class__.__name__]
        self.adding = True
        auxdict = {}
        if len(keys) == 1:
            exec("auxdict = self." + str(keys[0]))
        else:
            print "StorageObj " + str(self.name) + " has more than 1 dictionary, specify which one has to be used"
            raise KeyError
        auxdict[key] = auxdict[key] + other
        self.adding = False

    def __getitem__(self, key):
        keys = self.keyList[self.__class__.__name__]
        auxdict = {}
        if len(keys) == 1:
            exec("auxdict = self." + str(keys[0]))
        else:
            print "StorageObj " + str(self.name) + " has more than 1 dictionary, specify which one has to be used"
            raise KeyError
        # if (auxdict.types[str(auxdict.dict_name)] == 'counter'):
        #    #if self.adding == True:
        #    #    return 0
        #    return 0
        item = auxdict[key]
        return item

    def __setitem__(self, key, value):
        keys = self.keyList[self.__class__.__name__]
        auxdict = {}
        if len(keys) == 1:
            exec("auxdict = self." + str(keys[0]))
        else:
            print "StorageObj " + str(self.name) + " has more than 1 dictionary, specify which one has to be used"
            raise KeyError
        auxdict[key] = value

    def statistics(self):

        keys = self.keyList[self.__class__.__name__]

        reads = 0
        exec("reads = self." + str(keys[0]) + ".reads")
        writes = 0
        exec("writes = self." + str(keys[0]) + ".writes")

        if reads > 0 or writes > 0:
            print "####################################################"
            print "STATISTICS"
            print "Object:", self.__class__.__name__
            print "----------------------------------------------------"

        if reads > 0:
            if reads < 10:
                print "reads:                         ", reads
            else:
                if reads < 100:
                    print "reads:                        ", reads
                else:
                    if reads < 1000:
                        print "reads:                       ", reads
                    else:
                        if reads < 10000:
                            print "reads:                      ", reads
                        else:
                            print "reads:                     ", reads

            chits = 0
            exec("chits = self." + str(keys[0]) + ".cache_hits")
            if chits < 10:
                print "cache_hits(X):                 ", chits
            else:
                if chits < 100:
                    print "cache_hits(X):                ", chits
                else:
                    if chits < 1000:
                        print "cache_hits(X):               ", chits
                    else:
                        if chits < 10000:
                            print "cache_hits(X):              ", chits
                        else:
                            print "cache_hits(X):             ", chits
            pendreqs = 0
            exec("pendreqs = self." + str(keys[0]) + ".pending_requests")
            if pendreqs > 0:
                if pendreqs < 10:
                    print "pending_reqs:                  ", pendreqs
                else:
                    if pendreqs < 100:
                        print "pending_reqs:                 ", pendreqs
                    else:
                        if pendreqs < 1000:
                            print "pending_reqs:                ", pendreqs
                        else:
                            if pendreqs < 10000:
                                print "pending_reqs:               ", pendreqs
                            else:
                                print "pending_reqs:              ", pendreqs
            dbhits = 0
            exec("dbhits = self." + str(keys[0]) + ".miss")
            if dbhits < 10:
                print "miss(_):                       ", dbhits
            else:
                if dbhits < 100:
                    print "miss(_):                      ", dbhits
                else:
                    if dbhits < 1000:
                        print "miss(_):                     ", dbhits
                    else:
                        if dbhits < 10000:
                            print "miss(_):                    ", dbhits
                        else:
                            print "miss(_):                   ", dbhits

            cprefetchs = 0
            exec("cprefetchs = self." + str(keys[0]) + ".cache_prefetchs")
            if cprefetchs > 0:
                if cprefetchs < 10:
                            print "cprefetchs:                    ", cprefetchs
                else:
                    if cprefetchs < 100:
                            print "cprefetchs:                   ", cprefetchs
                    else:
                        if cprefetchs < 1000:
                            print "cprefetchs:                  ", cprefetchs
                        else:
                            if cprefetchs < 10000:
                                print "cprefetchs:                 ", cprefetchs
                            else:
                                print "cprefetchs:                ", cprefetchs

            cachepreffails = 0
            exec("cachepreffails = self." + str(keys[0]) + ".cache_prefetchs_fails")
            if cachepreffails < 10:
                print "cachepreffails:                ", cachepreffails
            else:
                if cachepreffails < 100:
                    print "cachepreffails:                 ", cachepreffails
                else:
                    print "cachepreffails:                ", cachepreffails

            cache_usage = 0
            if reads > 0:
                cache_usage = (float(chits) / float(reads)) * 100
                if cache_usage < 10:
                    print("cache_usage(cache hits/reads):  %.2f%%" % cache_usage)
                else:
                    if cache_usage < 100:
                        print("cache_usage(cache hits/reads): %.2f%%" % cache_usage)
                    else:
                        print("cache_usage(cache hits/reads):%.2f%%" % cache_usage)
            '''
            if cprefetchs > 0:
                used_prefetchs = (float(chits) / float(cprefetchs)) * 100
                if used_prefetchs < 10:
                    print("used_prefetchs (cache hits/cprefetchs):    %.2f%%" % used_prefetchs)
                else:
                    if used_prefetchs < 100:
                        print("used_prefetchs (cache hits/cprefetchs):   %.2f%%" % used_prefetchs)
                    else:
                        print("used_prefetchs (cache hits/cprefetchs):  %.2f%%" % used_prefetchs)
            '''
            '''
            if reads > 0:
                pendreqstotal = (float(pendreqs) / float(reads)) * 100
                if pendreqstotal < 10:
                    print("pending reqs(pendreqs/reads):   %.2f%%" % pendreqstotal)
                else:
                    if pendreqstotal < 100:
                        print("pending reqs(pendreqs/reads):  %.2f%%" % pendreqstotal)
                    else:
                        print("pending reqs(pendreqs/reads): %.2f%%" % pendreqstotal)
            '''

        if writes > 0:
            if writes < 10:
                print "writes:                        ", writes
            else:
                if writes < 100:
                    print "writes:                       ", writes
                else:
                    if writes < 1000:
                        print "writes:                      ", writes
                    else:
                        if writes < 10000:
                            print "writes:                     ", writes
                        else:
                            if writes < 100000:
                                print "writes:                    ", writes
                            else:
                                print "writes:                   ", writes

            cachewrite = 0
            exec("cachewrite = self." + str(keys[0]) + ".cachewrite")
            if cachewrite < 10:
                print "cachewrite:                    ", cachewrite
            else:
                if cachewrite < 100:
                    print "cachewrite:                   ", cachewrite
                else:
                    if cachewrite < 1000:
                        print "cachewrite:                  ", cachewrite
                    else:
                        if cachewrite < 10000:
                            print "cachewrite:                 ", cachewrite
                        else:
                            if cachewrite < 100000:
                                print "cachewrite:                ", cachewrite
                            else:
                                print "cachewrite:               ", cachewrite

            syncs = 0
            exec("syncs = self." + str(keys[0]) + ".syncs")
            if syncs < 10:
                print "syncs:                         ", syncs
            else:
                if syncs < 100:
                    print "syncs:                        ", syncs
                else:
                    if syncs < 1000:
                        print "syncs:                       ", syncs
                    else:
                        if syncs < 10000:
                            print "syncs:                      ", syncs
                        else:
                            if syncs < 100000:
                                print "syncs:                     ", syncs
                            else:
                                print "syncs:                    ", syncs

        if reads > 0 or writes > 0:
            print "------Times-----------------------------------------"
        if reads > 0:
            print "GETS"
            exec("cache_hits_time = self." + str(keys[0]) + ".cache_hits_time")
            print "cache_hits_time:               ", cache_hits_time
            cache_hits_time_med = 0.00000000000
            if chits > 0:
                cache_hits_time_med = cache_hits_time / chits
                print("cache_hits_time_med:            %.8f" % cache_hits_time_med)
            '''
            exec("pendreqsTimeRes = self." + str(keys[0]) + ".pending_requests_time_res")
            if pendreqsTimeRes < 10:
                print "pending_requests_time_res:     ", pendreqsTimeRes
            else:
                print "pending_requests_time_res:    ", pendreqsTimeRes
            pending_requests_time_med_res = 0.000000000000
            if pendreqs > 0:
                pending_requests_time_med_res = pendreqsTimeRes / pendreqs
                print("pending_requests_time_med_res:  %.8f" % pending_requests_time_med_res)
            '''
            '''
            exec("pendreqsTime = self." + str(keys[0]) + ".pending_requests_time")
            if pendreqsTime < 10:
                print "pending_requests_time:         ", pendreqsTime
            else:
                print "pending_requests_time:        ", pendreqsTime
            pending_requests_time_med = 0.000000000000
            if pendreqs > 0:
                pending_requests_time_med = pendreqsTime / pendreqs
                print("pending_requests_time_med:      %.8f" % pending_requests_time_med)
            '''
            '''
            pendreqfailstime = 0
            exec("pendreqfailstime = self." + str(keys[0]) + ".pending_requests_fails_time")
            if pendreqfailstime < 10:
                print "pendreqfailstime:              ", pendreqfailstime
            else:
                if pendreqfailstime < 100:
                    print "pendreqfailstime:               ", pendreqfailstime
                else:
                    print "pendreqfailstime:              ", pendreqfailstime
            '''
            exec("mtime = self." + str(keys[0]) + ".miss_time")
            if mtime < 10:
                print "miss_time:                     ", mtime
            else:
                print "miss_time:                    ", mtime
            miss_times_med = 0.000000000000
            if dbhits > 0:
                miss_times_med = mtime / dbhits
                print("miss_times_med:                 %.8f" % miss_times_med)

            #print "total_read_time:               ", str(cache_hits_time + pendreqsTime + mtime)
            print "total_read_time:               ", str(cache_hits_time + mtime)

        if writes > 0:
            print "WRITES"

            exec("syncstime = self." + str(keys[0]) + ".syncs_time")
            if syncstime < 10:
                print "syncs_time:                    ", syncstime
            else:
                print "syncs_time:                   ", syncstime
            syncs_times_med = 0.000000000000
            if syncs > 0:
                exec("syncs_times_med = (self." + str(keys[0]) + ".syncs_time / syncs)")
            print("syncs_times_med:                %.8f" % syncs_times_med)

            exec("cwritetime = self." + str(keys[0]) + ".cachewrite_time")
            if cwritetime < 10:
                print "cachewrite_time:               ", cwritetime
            else:
                print "cachewrite_time:              ", cwritetime
            cachewrite_times_med = 0.000000000000
            if cachewrite > 0:
                cachewrite_times_med = cwritetime / cachewrite
                print("cachewrite_times_med:           %.8f" % cachewrite_times_med)

            totalWritesTime = cwritetime + syncstime
            if totalWritesTime < 10:
                print "write_time:                    ", totalWritesTime
            else:
                print "write_time:                   ", totalWritesTime
            write_times_med = 0.00000000000
            if writes > 0:
                write_times_med = (totalWritesTime / writes)
                print("write_times_med:                %.8f" % write_times_med)

        if reads > 0:
            print "------Graph-----------------------------------------"
            exec("print self." + str(keys[0]) + ".cache_hits_graph")
        if reads > 0 or writes > 0:
            print "####################################################"
