import uuid


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
    pass


def end_task(params):
    """
    Terminates, if needed, the context (to save all data remaining in the batch) and the prefetch. It also prints
    the statistics of the StorageObjs if desired.
    Args:
        params: a list of objects (Blocks, StorageObjs, strings, ints, ...)
    """
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
    objid = str(objid)
    try:
        query = "SELECT * FROM hecuba.istorage WHERE storage_id = %s"
	
        
    except Exception as e:
        log.error("Query %s failed", query)
        raise e
    
    if not isinstance(objid, str):
		results = config.session.execute(query, [objid])[0]
    else:
		results = config.session.execute(query, [uuid.UUID(objid)])[0]

    log.debug("IStorage API:getByID(%s) of class %s", objid, results.class_name)
    return build_remotely(results._asdict())
