# author: G. Alomar
from cassandra.cluster import Cluster
from hecuba.dict import *
from hecuba.iter import *
from hecuba.datastore import *
from hecuba.storageobj import StorageObj
from conf.hecuba_params import execution_name
from conf.apppath import apppath
from app.qbeastiface import *
import glob

global qbeastInterface 
qbeastInterface= QbeastIface()

def classfilesparser():
    classes = {}
    filestoparse = glob.glob(apppath + "/app/*.py")
    for ftp in filestoparse:
        f = open(ftp, 'r')
        incomment = False
        classf = False
        classn = ''
        obj = []
        dict = 1
        for line in f:
            parsedline = line
            if '\n' in parsedline:
                parsedline = str(parsedline).replace('\n','')
            if "class " in parsedline:
                if "StorageObj" in parsedline:
                    classf = True
                    if not incomment:
                        parsedline = parsedline.split(" ")
                        parsedline = parsedline[1]
                        parsedline = parsedline.split("(")
                        classn = parsedline[0]
            if "'''" in parsedline:
                if classf:
                    if not incomment:
                        incomment = True
                    else:
                        incomment = False
            # example1:"@ClassField fieldname dictionartype <<k1:int,k2:int>,tuple<v1:int,v2:float,v3:text>>" #still not done
            # example1:"@ClassField fieldname dictionartype <<k1:int,k2:int>,v1:int,v2:int>" #done
            # example2:"@ClassField fieldname elementaltype" #done
            # there can't be spaces in the signature, as parsing becomes much more complicated
            if "@ClassField" in parsedline:
                if incomment:
                    parsedline = parsedline.split("@")
                    parsedline = str(parsedline[1]).split(" ")
                    parname = str(parsedline[1])
                    partype = str(parsedline[2])
                    if len(parsedline) >= 4:
                        parvaltypes = str(parsedline[3])
                    else:
                        parvaltypes = '-'
                    if partype == 'dict':
                        dictsign = str(parsedline[3])[1:len(str(parsedline[3]))-1]
                        dictkey = dictsign.split(">")[0]
                        dictkey = str(dictkey)[1:len(str(dictkey))]
                        if "," in dictkey:
                            for k in dictkey.split(","):
                                obj.append([k.split(":")[0], k.split(":")[1], "y", dict])
                        else:
                            obj.append([dictkey.split(":")[0], dictkey.split(":")[1], "y", dict])
                        dictval = dictsign.split(">")[1]
                        dictval = str(dictval)[1:len(str(dictval))]
                        if "," in dictval:
                            for v in dictval.split(","):
                                obj.append([v.split(":")[0], v.split(":")[1], "n", dict])
                        else:
                            obj.append([dictval.split(":")[0], dictval.split(":")[1], "n", dict])
                        dict += 1
                    else:
                        obj.append([parname, partype, parvaltypes])
        if not classn == '':
            classes[classn] = obj
    return classes



cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)
session = cluster.connect()

keyspace = execution_name
keyspaceaux = 'config' + keyspace

repl_factor = "3"
repl_class = "SimpleStrategy"

try:
    session.execute("CREATE KEYSPACE IF NOT EXISTS " + keyspace + " WITH REPLICATION = { 'class' : \'" + repl_class + "\', 'replication_factor' : " + repl_factor + " };")
except Exception as e:
    print "Cannot create keyspace", e

try:
    session.execute("CREATE KEYSPACE IF NOT EXISTS " + keyspaceaux + " WITH REPLICATION = { 'class' : \'" + repl_class + "\', 'replication_factor' : " + repl_factor + " };")
except Exception as e:
    print "Cannot create keyspace", e

try:
    session.execute("CREATE TABLE IF NOT EXISTS " + keyspaceaux + ".attribs (dictName text, dataName text, dataType text, dataValue text, PRIMARY KEY (dictName, dataName));")
except Exception as e:
    print "Warning: Object attribs cannot be created in persistent storage.", e

session.set_keyspace(keyspaceaux)

KeyIter.blockkeyspace = keyspace
PersistentDict.keyspace = keyspace
cluster2 = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)
PersistentDict.session = cluster2.connect(keyspace)

classes1 = classfilesparser()

conversion = {'atomicint': 'counter',
              'str': 'text',
              'bool': 'boolean',
              'decimal': 'decimal',
              'float': 'double',
              'int': 'int',
              'tuple': 'list',
              'list': 'list',
              'generator': 'list',
              'frozenset': 'set',
              'set': 'set',
              'dict': 'map',
              'long': 'bigint',
              'buffer': 'blob',
              'bytearray': 'blob',
              'counter': 'counter'}

classind = 1
dicts = {}
for Class in classes1:
    dictName1 = Class
    classData = classes1[Class]
    for value in classData:
        if value[2] == "y" or value[2] == "n":
            try:
                session.execute("CREATE TABLE IF NOT EXISTS " + keyspaceaux + ".\"" + dictName1 + "\" (dictID text, dataName text, isKey text, keyOrder text, dataType text, PRIMARY KEY (dictID, dataName));")
            except Exception as e:
                print "Warning: Object", dictName1, "cannot be created in persistent storage.", e
    for value in classData:
        keyind = 1
        valtype = conversion[value[1]]
        PersistentDict.types[str(value[0])] = str(valtype)
        if value[2] == 'y' or value[2] == 'n':
            if value[2] == 'y':
                try:
                    session.execute("INSERT INTO " + keyspaceaux + ".\"" + dictName1 + "\"(dictID, dataName, isKey, keyOrder, dataType) VALUES ( %s, %s, %s, %s, %s)", (str(value[3]), str(value[0]), "yes", str(keyind), str(valtype)))
                except Exception as e:
                    print "Error: Object", str(value[0]), "cannot be inserted in persistent storage.", e

                keyind += 1
                StorageObj.keyList[dictName1].append(value[0])
            if value[2] == 'n':
                try:
                    session.execute("INSERT INTO " + keyspaceaux + ".attribs(dictName, dataName, dataType, dataValue) VALUES ( %s, %s, %s, %s)", (str(dictName1), str(value[0]), "dict", str(keyspace)+'.\"'+str(dictName1)+'\"'))
                except Exception as e:
                    print "Error: Object", str(value[0]), "cannot be inserted in persistent storage.", e
                try:
                    session.execute("INSERT INTO " + keyspaceaux + ".\"" + dictName1 + "\"(dictID, dataName, dataType) VALUES ( %s, %s, %s)", (str(value[3]), str(value[0]), str(valtype)))
                except Exception as e:
                    print "Error: Object", str(value[0]), "cannot be inserted in persistent storage.", e

                keyind += 1
        else:
            query2 = "SELECT * FROM " + keyspaceaux + ".attribs WHERE dictname = \'" + str(dictName1) + "\' AND dataName = \'" + str(value[0]) + "\';"
            try:
                result = session.execute(query2)
            except Exception as e:
                print "error:", e
            if len(result) == 0:
                if 'list' in str(valtype):
                    try:
                        session.execute("INSERT INTO " + keyspaceaux + ".attribs(dictName, dataName, dataType, dataValue) VALUES ( %s, %s, %s, %s)", (str(dictName1), str(value[0]), str(valtype)+'_'+str(value[2]), str(keyspace)+'.\"'+str(dictName1)+str(value[0])+'\"'))
                    except Exception as e:
                        print "Error: Object", str(value[0]), "cannot be inserted in persistent storage.", e
                    querytable = "CREATE TABLE " + execution_name + ".\"" + str(dictName1) + str(value[0]) + "\" (position int, type text, value text, PRIMARY KEY (position));"
                    try:
                        session.execute(querytable)
                    except Exception as e:
                        print "Object", dictName1, "cannot be created in persistent storage", e
                else:
                    try:
                        session.execute("INSERT INTO " + keyspaceaux + ".attribs(dictName, dataName, dataType, dataValue) VALUES ( %s, %s, %s, '-')", (str(dictName1), str(value[0]), str(valtype)))
                    except Exception as e:
                        print "Error: Object", str(value[0]), "cannot be inserted in persistent storage.", e

session.shutdown()
cluster.shutdown()
