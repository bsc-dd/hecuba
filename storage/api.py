# author: G. Alomar
from cassandra.cluster import Cluster
from hecuba.datastore import *
from hecuba.iter import Block
from hecuba.dict import *
from hecuba.storageobj import *
from hecuba.storageobjix import *
import collections

from conf.hecuba_params import *
from conf.apppath import apppath


def start_task(params):
    if not 'prefetch_activated' in globals():
        global prefetch_activated
        prefetch_activated = True
    if not 'batch_activated' in globals():
        global batch_activated
        batch_activated = True
    if prefetch_activated or batch_activated:
        path = apppath + '/conf/imports.py'
        file = open(path, 'r')
        for line in file:
            exec line

    if batch_activated:
        for param in params:
            if isinstance(param, StorageObj) or isinstance(param, StorageObjIx) or isinstance(param, Block):
                param.cntxt = context(param)
                param.cntxt.__enter__()

def end_task(params):
    if not 'prefetch_activated' in globals():
        global prefetch_activated
        prefetch_activated = True
    if not 'batch_activated' in globals():
        global batch_activated
        batch_activated = True
    if prefetch_activated or batch_activated:
        path = apppath + '/conf/imports.py'
        file = open(path, 'r')
        for line in file:
            exec line

    if batch_activated:
        for param in params:
            if isinstance(param, Block) or isinstance(param, StorageObj) or isinstance(param, StorageObjIx):
                try:
                    param.cntxt.__exit__()
                except Exception as e:
                    print "error trying to exit context:", e

    if prefetch_activated:
        for param in params:
            if isinstance(param, Block):
                keys = param.storageobj.keyList[param.storageobj.__class__.__name__]
                exec("persistentdict = param.storageobj." + str(keys[0]))
                if persistentdict.prefetch == True:
                    try:
                        persistentdict.end_prefetch()
                    except Exception as e:
                        print "error trying to prefetch:", e

    if not 'statistics_activated' in globals():
        global statistics_activated
        statistics_activated = True
    if statistics_activated:
        for param in params:
            if isinstance(param, Block):
                param.storageobj.statistics()
            if isinstance(param, StorageObj) or isinstance(param, StorageObjIx):
                param.statistics()

def getByID(objid):
    path = apppath + '/conf/imports.py'
    file = open(path, 'r')
    for line in file:
        exec(line)
    objidsplit = objid.split("_")
    print "objidsplit:", objidsplit

    if len(objidsplit) == 2:
        result = ''
        exec('result = ' + objidsplit[0] + '()')
        print "str(objidsplit[0]):", str(objidsplit[0])
        result.getByName(str(objidsplit[0]))
        return result

    else:
        cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)
        session = cluster.connect()
        #(ksp,tab,dict_name,obj_type,entryPoint,port,tokens) = session.execute("SELECT ksp,tab,dict_name,obj_type,entrypoint,port,tkns FROM hecuba.blocks WHERE blockid = %s",(objid))[0]
        ksp =        session.execute("SELECT ksp FROM hecuba.blocks WHERE blockid = %s",(objid,))[0].ksp
        tab =        session.execute("SELECT tab FROM hecuba.blocks WHERE blockid = %s",(objid,))[0].tab
        dict_name =  session.execute("SELECT dict_name FROM hecuba.blocks WHERE blockid = %s",(objid,))[0].dict_name
        obj_type =   session.execute("SELECT obj_type FROM hecuba.blocks WHERE blockid = %s",(objid,))[0].obj_type
        entryPoint = session.execute("SELECT entrypoint FROM hecuba.blocks WHERE blockid = %s",(objid,))[0].entrypoint
        port =       session.execute("SELECT port FROM hecuba.blocks WHERE blockid = %s",(objid,))[0].port
        tokens =     session.execute("SELECT tkns FROM hecuba.blocks WHERE blockid = %s",(objid,))[0].tkns
        if str(obj_type) == 'qbeast':
            print "indexed storageobj"
            blockid = objid
            metadata = cluster.metadata
            tokenmap = metadata.token_map.token_to_host_owner
            odtokenmap = collections.OrderedDict(sorted(tokenmap.items()))
            if not 'prefetch_activated' in globals():
                global prefetch_activated
                prefetch_activated = True
            for position in tokens: 
                for key, val in odtokenmap.iteritems():
                    # (self, peer,        keynames,           tablename,     blockkeyspace, myuuid)
                    b = IxBlock(str(val), tab,                tab,           ksp,           objid)
                    exec("b.storageobj = " + str(dict_name) + "('" + str(dict_name) + "')")
                    if prefetch_activated:
                        b.storageobj.init_prefetch(b)
                    session.shutdown()
                    cluster.shutdown()
                    return b
        else:
            print "normal storageobj"
            # blockKeyspace = objidsplit[0]
            # blockKeyNames = str(objidsplit[1])
            # blockTableName = objidsplit[2]
            # blockRange = objidsplit[3]
            blockranges = objidsplit[3:len(objidsplit)]
            blockrangesf = ''
            for ind, val in enumerate(blockranges):
                if ind < (len(blockranges)-1):
                    blockrangesf += str(val) + '_'
                else:
                    blockrangesf += str(val)
            cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)
            session = cluster.connect()
            metadata = cluster.metadata
            tokenmap = metadata.token_map.token_to_host_owner
            odtokenmap = collections.OrderedDict(sorted(tokenmap.items()))
            position = 0
            if not 'prefetch_activated' in globals():
                global prefetch_activated
            prefetch_activated = True
            for key, val in odtokenmap.iteritems():
                if str(position) == objidsplit[3]:
                    b = Block((str(val), blockrangesf), str(objidsplit[1]), objidsplit[2], objidsplit[0])
                    exec("b.storageobj = " + str(objidsplit[2]) + "('" + str(objidsplit[2]) + "')")
                    if prefetch_activated:
                        b.storageobj.init_prefetch(b)
                    session.shutdown()
                    cluster.shutdown()
                    return b
                position += 1
