from storage.cql_iface.tests.mockIStorage import IStorage


class StorageObj(IStorage):
    """
    This class is where information will be stored in Hecuba.
    The information can be in memory, stored in a python dictionary or local variables, or saved in a
    DB(Cassandra), depending on if it's persistent or not.
    """

    def __new__(cls, *args, name='', **kwargs):
        toret = super(StorageObj, cls).__new__(cls)

        return toret

    def __init__(self, *args, **kwargs):
        """
            Creates a new storageobj.
            Args:
                name (string): the name of the Cassandra Keyspace + table where information can be found
                tokens (list of tuples): token ranges assigned to the new StorageObj
                storage_id (string):  an unique storageobj identifier
                istorage_props dict(string,string): a map with the storage id of each contained istorage object.
                kwargs: more optional parameters
        """
        # Assign private attributes
        # if self._is_persistent:
        #    self._load_attributes()
        super(StorageObj, self).__init__()
