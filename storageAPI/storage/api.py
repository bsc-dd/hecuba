

def init(config_file_path=None):
    """
    Executed by Master
    Function that can be useful when running the application with COMPSs >= 2.0
    It is executed at the beginning of the application
    """
    pass


def finish():
    """
    Executed by master
    Function that can be useful when running the application with COMPSs >= 2.0
    It is executed at the end of the application
    """
    pass


def initWorker(config_file_path=None):
    """
    Executed by Each java Worker before spawning the executors
    Function that can be useful when running the application with COMPSs >= 2.0
    It is executed at the beginning of the application
    """
    pass


def finishWorker():
    """
    Executed by each python Worker before ending
    Function that can be useful when running the application with COMPSs >= 2.0
    It is executed at the end of the application
    """
    pass

def initWorkerPostFork():
    """
    Executed by each executor at the beginning of their execution
    """
    pass

def finishWorkerPostFork():
    """
    Executed by each executor before ending
    """
    pass

def start_task(params):
    """
    Initializes, if needed, the global vars for prefetch and batch, and starts the context if batch is activated
    Args:
        params: a list of objects (Blocks, StorageObjs, strings, ints, ...)
    """
    #f = open("/home/bsc31/bsc31226/compssDEBUG.txt", "a")
    #f.write("JCOSTA ENTER =================\n")
    #f.write("params      = {}\n".format([(i.name, i.direction,type(i.content)) for i in params]))
    #f.write("JCOSTA ENTER =================\n")
    #f.close()
    pass


def end_task(params):
    """
    Terminates, if needed, the context (to save all data remaining in the batch) and the prefetch. It also prints
    the statistics of the StorageObjs if desired.
    Args:
        params: a list of objects (Blocks, StorageObjs, strings, ints, ...)
    """
    import builtins;

    #f = open("/home/bsc31/bsc31226/compssDEBUG.txt", "a")
    ##print("JCOSTA =================", flush=True)
    ##print("params_type = {}".format(type(params)), flush=True)
    ##print("params_len  = {}".format(len(params)), flush=True)
    ##print("params      = {}".format([type(i) for i in params]), flush=True)
    ##print(builtins.__dir__(), flush=True)
    ##print("JCOSTA =================", flush=True)
    #f.write("JCOSTA =================\n")
    #f.write("params_type = {}\n".format(type(params)))
    #f.write("params_len  = {}\n".format(len(params)))
    #f.write("params      = {}\n".format([i for i in params]))
    #f.write("params      = {}\n".format([(i.name, i.direction,type(i.content)) for i in params]))
    #f.write("builtins dir = {}\n".format(builtins.__dir__()))
    #f.write("JCOSTA =================\n")
    #f.close()
    pass


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
    We rebuild the object from its id.

    Args:
        objid (str):  object identifier

    Returns:
         Hecuba Object

    """
    """
               TODO
               Args:
                   objid (str):  object identifier
               Returns:
                    (Block| Storageobj)
               """
    from hecuba import log
    from hecuba.IStorage import build_remotely
    from hecuba import config
    from hecuba import StorageNumpy, StorageDict
    from hecuba import StorageObj as StorageObject
    import uuid

    query = "SELECT * FROM hecuba.istorage WHERE storage_id = %s"

    if isinstance(objid, str):
        objid = uuid.UUID(objid)

    results = config.session.execute(query, [objid])
    if not results:
        raise RuntimeError("Object {} not found on hecuba.istorage".format(objid))

    results = results[0]

    log.debug("IStorage API:getByID(%s) of class %s", objid, results.class_name)
    return build_remotely(results._asdict())
