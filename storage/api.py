# author: G. Alomar

from hecuba.iter import Block
from hecuba.dict import *
from hecuba.storageobj import *
from hecuba import config


def init(config_file_path=None):
    """
    Function that can be useful when running the application with COMPSs >= 2.0
    It is executed at the beginning of the application
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
    if config.batch_size > 1:
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

    if config.batch_size > 1:
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
                    persistent_dict = param.storageobj._get_default_dict()
                    if persistent_dict.prefetch:
                        persistent_dict.end_prefetch()

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
        objid (str):  object identifier

    Returns:
         (Block| Storageobj)

    """
    objidsplit = objid.split("_")

    if len(objidsplit) == 2:
        objid = objidsplit[0]


    if len(objidsplit) == 2:
        results = config.session.execute("SELECT * FROM hecuba.storage_objs WHERE object_id = %s", (objid,))[0]
        class_name = results.class_name
    else:
        results = config.session.execute("SELECT * FROM hecuba.blocks WHERE blockid = %s", (objid,))[0]
        class_name = results.class_name
    last = 0
    for key, i in enumerate(class_name):
        if i == '.' and key > last:
            last = key
    module = class_name[:last]
    cname = class_name[last + 1:]
    mod = __import__(module, globals(), locals(), [cname], 0)
    b = getattr(mod, cname).build_remotely(results)

    if len(objidsplit) == 1:
        if config.prefetch_activated and b.supportsPrefetch:
            b.storageobj.init_prefetch(b)
    return b
