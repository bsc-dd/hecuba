# author: G. Alomar

from hecuba.iter import Block
from hecuba.dict import *
from hecuba.storageobj import *
from hecuba import config
import collections
import hecuba



def init(config_file_path=None):
    """
    Function that can be useful when running the application with COMPSs >= 2.0
    It is executed at the begining of the application
    """
    pass


def finish():
    """
    Function that can be useful when running the application with COMPSs >= 2.0
    It is executed at the end of the application
    """
    pass


def start_task(params):
    """
    Initializes, if needed, the global vars for prefetch and batch, and starts the context if batch is activated
    Args:
        params: a list of objects (Blocks, StorageObjs, strings, ints, ...)
    """
    if type(params) is not list:
        raise ValueError('call start_task with a list of params')
    if config.batch_activated:
        for param in params:
            if issubclass(param.__class__, StorageObj) or issubclass(param.__class__, Block) and param.needContext:
                param.cntxt = context(param)
                param.cntxt.__enter__()


def end_task(params):
    """
    Terminates, if needed, the context (to save all data remaining in the batch) and the prefetch. It also prints
    the statistics of the StorageObjs if desired.
    Args:
        params: a list of objects (Blocks, StorageObjs, strings, ints, ...)
    """

    if config.batch_activated:
        for param in params:
            if hasattr(param, 'needContext') and param.needContext:
                try:
                    param.cntxt.__exit__()
                except Exception as e:
                    print "error trying to exit context:", e

    if config.prefetch_activated:
        for param in params:
            if hasattr(param, 'needContext') and param.needContext:
                if issubclass(param.__class__, Block):
                    keys = param.storageobj.keyList[param.storageobj.__class__.__name__]
                    exec ("persistentdict = param.storageobj." + str(keys[0]))
                    if persistentdict.prefetch == True:
                        try:
                            persistentdict.end_prefetch()
                        except Exception as e:
                            print "error trying to prefetch:", e

    if config.statistics_activated:
        for param in params:
            if issubclass(param.__class__, Block) and param.supportsStatistics:
                param.storageobj.statistics()
            if issubclass(param.__class__, StorageObj):
                param.statistics()


def getByID(objid):
    """
    We rebuild the object from its id. The id can either be:
    block: UUID (eg. f291f008-a520-11e6-b42e-5b582e04fd70)
    storageobj: UUID_(version) (eg. f291f008-a520-11e6-b42e-5b582e04fd70_1)
    Args:
        objid: str object identifier

    Returns: (Block| Storageobj)

    """
    objidsplit = objid.split("_")

    if len(objidsplit) == 2:
        objid = objidsplit[0]

    try:
        results = session.execute("SELECT * FROM hecuba.blocks WHERE blockid = %s", (objid,))[0]

        if len(objidsplit) == 2:
            classname = results.storageobj_classname
        else:
            classname = results.block_classname
        last = 0
        for key, i in enumerate(classname):
            if i == '.' and key > last:
                last = key
        module = classname[:last]
        cname = classname[last + 1:]
        exec ('from %s import %s' % (module, cname))
        exec ('obj_class = ' + cname)

        b = obj_class.build_remotely(results)

        if len(objidsplit) == 1:
            #This runs only if it is a block
            if not 'prefetch_activated' in globals():
                global prefetch_activated
                prefetch_activated = True
            if prefetch_activated and b.supportsPrefetch:
                b.storageobj.init_prefetch(b)
        return b
    except Exception as e:
        print "getByID error:", e
        raise e
