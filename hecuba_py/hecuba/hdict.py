import uuid
from typing import NamedTuple

import storage
from storage.cql_iface import log
from .IStorage import IStorage, AlreadyPersistentError
from .tools import build_remotely, storage_id_from_name, transform_to_dm



class StorageDict(IStorage):
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
                #{"type": StorageDict, "value_id": {"k1": int}, "fields": {"a": int, "b": str, "c": float}}
                cls._data_model_def = dict()
                cls._data_model_def['type'] = cls
                cls._data_model_def['value_id'] = {'k': uuid.UUID}
                cls._data_model_def['fields'] = {k: v for k, v in args[0].__annotations__.items()}
                cls._data_model_id = storage.StorageAPI.add_data_model(cls._data_model_def)

            # Storage data model
            # keys = {k: uuid.UUID if issubclass(v, IStorage) else v for k, v in cls._data_model_def["value_id"]}
            # cols = {k: uuid.UUID if issubclass(v, IStorage) else v for k, v in cls._data_model_def["cols"]}

            cls._data_model_id = storage.StorageAPI.add_data_model(cls._data_model_def)

        toret = super(StorageDict, cls).__new__(cls, kwargs)
        storage_id = kwargs.get('storage_id', None)

        if storage_id is None and name:
            storage_id = storage_id_from_name(name)

        if name or storage_id:
            toret.setID(storage_id)
            toret.set_name(name)
            toret._is_persistent = True
            storage.StorageAPI.register_persistent_object(cls._data_model_id, toret)
        return toret

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def __iter__(self):
        """
        Method that overloads the python dict basic iteration, which returns
        an iterator over the dictionary keys.
        """
        return self.keys

    def make_persistent(self, name):
        """
        Method to transform a StorageDict into a persistent object.
        This will make it use a persistent DB as the main location
        of its data.
        Args:
            name:
        """
        if self._is_persistent:
            raise AlreadyPersistentError("This StorageDict is already persistent {}", name)

        # Update local StorageDict metadata
        super().make_persistent(name)

        storage.StorageAPI.register_persistent_object(self.__class__._data_model_id, self)

        keys = []
        values = []
        # Storing all in-memory values to cassandra
        for i, (key, value) in enumerate(dict.items(self)):
            keys.append(key)
            if isinstance(value, IStorage):
                if not value._is_persistent:
                    sd_name = name + '_' + i
                    value.make_persistent(sd_name)
                values.append(value.getID())
            else:
                values.append(value)

        storage.StorageAPI.put_records(self.storage_id, keys, values)

        super(StorageDict, self).clear()

    def stop_persistent(self):
        """
        Method to turn a StorageDict into non-persistent.
        """
        log.debug('STOP PERSISTENCE')
        for obj in self.values():
            if isinstance(obj, IStorage):
                obj.stop_persistent()

        super().stop_persistent()

    def __delitem__(self, key):
        """
        Method to delete a specific entry in the dict in the key position.
        Args:
            key: position of the entry that we want to delete
        """
        if not self.storage_id:
            dict.__delitem__(self, key)
        else:
            storage.StorageAPI.put_records(self.storage_id, [key], [])

    def __getitem__(self, key):
        """
        If the object is persistent, each request goes to the hfetch.
        Args:
             key: the dictionary key
        Returns
             item: value found in position key
        """
        log.debug('GET ITEM %s', key)

        if not self.storage_id:
            return dict.__getitem__(self, key)

        if not isinstance(key, list):
            key = [key]

        # Returns always a list with a single entry for the key
        keys = NamedTuple('keys', [('k', int)])
        keys = keys(self.storage_id)._asdict()
        persistent_result = storage.StorageAPI.get_record(self.storage_id, keys)
        index = list(self._data_model_def["fields"]).index(key[0])
        object.__setattr__(self, key[0], persistent_result[index])
        return persistent_result[index]

        # we need to transform UUIDs belonging to IStorage objects and rebuild them
        # final_results = []
        #
        # for i, element in enumerate(persistent_result):
        #     col_type = self.__class__._data_model_def['cols'][i]
        #     if issubclass(col_type, IStorage):
        #         # element is not a built-in type
        #         table_name = self.storage_id + '_' + str(key)
        #         info = {"name": table_name, "storage_id": element, "class_name": col_type}
        #         element = build_remotely(info)
        #
        #     final_results.append(element)
        #
        # return final_results

    def __setitem__(self, key, val):
        """
           Method to insert values in the StorageDict
           Args:
               key: the position of the value that we want to save
               val: the value that we want to save in that position
        """
        log.debug('SET ITEM %s->%s', key, val)

        if not isinstance(val, list):
            val = [val]

        #keys = [def_type(val[i]) for i, def_type in enumerate(self._data_model_def['value_id'].values())]
        #vals = [def_type(val[i]) for i, def_type in enumerate(self._data_model_def['cols'].values())]

        if not self._is_persistent:
            dict.__setitem__(self, key, val[0])
        else:

            keys = NamedTuple('keys', [('k1', int)])
            keys = keys(self.storage_id)._asdict()

            fields = NamedTuple('fields', [(key, type(val[0]))])
            fields = fields(val[0])._asdict()

            storage.StorageAPI.put_record(self.storage_id, keys, fields)

    def __repr__(self):
        """
        Overloads the method used by print to show a StorageDict
        Returns: The representation of the data stored in the StorageDict

        """
        to_return = {}
        for item in self.items():
            to_return[item[0]] = item[1]
            if len(to_return) == 20:
                return str(to_return)
        if len(to_return) > 0:
            return str(to_return)
        return ""
