from storage.cql_iface.tests.mockIStorage import IStorage
from storage.cql_iface.tests.mocktools import storage_id_from_name, transform_to_dm
import uuid


class StorageObj(IStorage):
    """
    This class is where information will be stored in Hecuba.
    The information can be in memory, stored in a python dictionary or local variables, or saved in a
    DB(Cassandra), depending on if it's persistent or not.
    """

    def __new__(cls, name='', *args, **kwargs):
        if not cls._data_model_id:
            # User data model
            keys = {}
            try:
                cls._data_model_def = kwargs['data_model']
            except KeyError:
                pass
                cls._data_model_def = dict()
                cls._data_model_def["cols"] = transform_to_dm(cls)
                cls._data_model_def['value_id'] = {'storage_id': uuid.UUID}
                cls._data_model_def['type'] = cls

        toret = super(StorageObj, cls).__new__(cls)
        storage_id = kwargs.get('storage_id', None)
        if storage_id is None and name:
            storage_id = storage_id_from_name(name)

        if name or storage_id:
            toret.setID(storage_id)
            toret.set_name(name)
            toret._is_persistent = True
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