from storage.cql_iface.tests.mockIStorage import IStorage
from storage.cql_iface.tests.mocktools import build_remotely, storage_id_from_name, transform_to_dm
import storage


class StorageDict(IStorage, dict):
    # """
    # Object used to access data from workers.
    # """

    def __new__(cls, name='', *args, **kwargs):
        """
        Creates a new StorageDict.

        Args:
            name (string): the name of the collection/table (keyspace is optional)
            storage_id (string): the storage id identifier
            args: arguments for base constructor
            kwargs: arguments for base constructor
        """
        if not cls._data_model_id:
            # User data model
            keys = {}
            try:
                cls._data_model_def = kwargs['data_model']
            except KeyError:
                import typing
                dms = []
                for ob in cls.__orig_bases__:
                    if isinstance(ob, typing.GenericMeta):
                        dms.append(transform_to_dm(ob))
                if len(dms) != 1:
                    raise ValueError("Different orig bases than expected ({})".format(len(dms)))

                cls._data_model_def = dms[0]
                cls._data_model_def['type'] = cls

            # Storage data model
            #keys = {k: uuid.UUID if issubclass(v, IStorage) else v for k, v in cls._data_model_def["value_id"]}
            #cols = {k: uuid.UUID if issubclass(v, IStorage) else v for k, v in cls._data_model_def["cols"]}

            cls._data_model_id = storage.StorageAPI.add_data_model(cls._data_model_def)

        toret = super(StorageDict, cls).__new__(cls, kwargs)
        storage_id = kwargs.get('storage_id', None)

        if storage_id is None and name:
            storage_id = storage_id_from_name(name)

        if name or storage_id:
            toret.setID(storage_id)
            toret.set_name(name)
            toret._is_persistent = True
            #storage.StorageAPI.register_persistent_object(cls._data_model_id, toret)
        return toret

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

