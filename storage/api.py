# author: G. Alomar

from hecuba.iter import Block
from hecuba.dict import *
from hecuba.storageobj import *
from hecuba.settings import *
import collections
import hecuba

from conf.hecuba_params import *


def start_task(params):
    if type(params) is not list:
        raise ValueError('call start_task with a list of params')
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
            if issubclass(param.__class__, StorageObj) or issubclass(param.__class__, Block) and param.needContext:
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
            if hasattr(param,'needContext') and param.needContext:
                try:
                    param.cntxt.__exit__()
                except Exception as e:
                    print "error trying to exit context:", e

    if prefetch_activated:
        for param in params:
            if hasattr(param,'needContext') and param.needContext:
                if issubclass(param.__class__, Block):
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
            if issubclass(param.__class__, Block):
                param.storageobj.statistics()
            if issubclass(param.__class__, StorageObj):
                param.statistics()

def getByID(objid):
    path = apppath + '/conf/imports.py'
    file = open(path, 'r')
    for line in file:
        exec(line)
    objidsplit = objid.split("_")

    if len(objidsplit) == 2:
        result = ''
        exec('result = ' + objidsplit[0] + '()')
        result.getByName(str(objidsplit[0]))
        return result

    else:

        try:
            # exec ("b = %d()"$(class_name))

            (blockid, classname, tkns, entryPoint, port, ksp, tab, dict_name, obj_type) = \
            session.execute(
                "SELECT blockid, classname, tkns, entryPoint ,port, ksp, tab, dict_name, obj_type " +
                "FROM hecuba.blocks WHERE blockid = %s", (objid,))[0]

            last=0
            for key, i in enumerate(classname):
                if i == '.' and key > last:
                    last = key
            module=classname[:last]
            cname=classname[last+1:]
            exec('from %s import %s'%(module,cname))
            exec('block_class = '+cname)

            b = block_class.build_remotely(blockid, classname, tkns, entryPoint, port, ksp, tab, dict_name, obj_type)

            if not 'prefetch_activated' in globals():
                global prefetch_activated
                prefetch_activated = True
            if prefetch_activated and b.supportsPrefetch:
                b.storageobj.init_prefetch(b)
            return b
        except Exception as e:
            print "Error:", e

