from storage.cql_iface.tests.mockIStorage import IStorage

class StorageDict(IStorage, dict):
    # """
    # Object used to access data from workers.
    # """

    def __new__(cls,  *args, name='', **kwargs):
        """
        Creates a new StorageDict.

        Args:
            name (string): the name of the collection/table (keyspace is optional)
            storage_id (string): the storage id identifier
            args: arguments for base constructor
            kwargs: arguments for base constructor
        """
        toret = super(StorageDict, cls).__new__(cls, kwargs)
        return toret

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

