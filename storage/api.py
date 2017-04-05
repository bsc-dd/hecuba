# author: G. Alomar

from hecuba.storageobj import *
from hecuba import config, log


def initWorker(config_file_path=None):
    """
    Function that can be useful when running the application with COMPSs >= 2.0
    It is executed at the beginning of the application
    """
    pass


def finishWorker():
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
            if issubclass(param.__class__, StorageObj) or issubclass(param.__class__, StorageDict) and param._needContext:
                param._cntxt = context(param)
                param._cntxt.__enter__()


def end_task(params):
    """
    Terminates, if needed, the context (to save all data remaining in the batch) and the prefetch. It also prints
    the statistics of the StorageObjs if desired.
    Args:
        params: a list of objects (Blocks, StorageObjs, strings, ints, ...)
    """
    if config.batch_size > 1:
        for param in params:
            if hasattr(param, '_needContext') and param._needContext:
                try:
                    param._cntxt.__exit__()
                except Exception as e:
                    print "error trying to exit context:", e


class TaskContext(object):
    def __init__(self, logger, values, **kwargs):
        self.logger = logger
        self.values = values

    def __enter__(self):
        # Do something prolog
        start_task(self.values)
        # Ready to start the task
        self.logger.info("Prolog finished")
        pass

    def __exit__(self, type, value, traceback):
        # Do something epilog
        end_task(self.values)
        # Finished
        self.logger.info("Epilog finished")
        pass

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

    results = config.session.execute("SELECT * FROM hecuba.istorage WHERE storage_id = %s", (objid,))[0]
    class_name = results.class_name

    log.debug("Storage API:getByID(%s) of class %s", objid, class_name)
    last = 0
    for key, i in enumerate(class_name):
        if i == '.' and key > last:
            last = key
    module = class_name[:last]
    cname = class_name[last + 1:]
    mod = __import__(module, globals(), locals(), [cname], 0)
    b = getattr(mod, cname).build_remotely(results)
    b._storage_id = objid
    return b
