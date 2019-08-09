from . import log, Parser
from .storageiter import StorageIter

from .IStorage import IStorage, AlreadyPersistentError
from .tools import build_remotely, storage_id_from_name, update_type
import storage
import uuid


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
                persistent_props = Parser("TypeSpec").parse_comments(cls.__doc__)
                keys = {k: update_type(v) for k, v in persistent_props['primary_keys']}
                cols = {k: update_type(v) for k, v in persistent_props['columns']}
                cls._data_model_def = {"type": cls.__name__, 'keys': keys, 'cols': cols}

            # Storage data model
            keys = {k: uuid.UUID if issubclass(v, IStorage) else v for k, v in cls._data_model_def["keys"].items()}
            cols = {k: uuid.UUID if issubclass(v, IStorage) else v for k, v in cls._data_model_def["cols"].items()}

            cls._data_model_id = storage.StorageAPI.add_data_model({"type": cls.__name__, 'keys': keys, 'cols': cols})

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

    def __contains__(self, key):
        """
        Method that checks if a given key exists in a StorageDict.
        Args:
            key: the position that we want to check if exists.
        Returns:
            boolean (true - exists, false - doesn't exist).
        """
        if not self._is_persistent:
            return dict.__contains__(self, key)

        return storage.StorageAPI.get_records(self.storage_id, [key]) != []

    def keys(self):
        """
        This method return a list of all the keys of the StorageDict.
        Returns:
          list: a list of keys
        """
        iter_cols = self._data_model_def.get('keys', None)
        iter_model = {"type": "StorageIter", "name": self.get_name(), "cols": iter_cols}
        if self.storage_id:
            return StorageIter(storage_id=self.storage_id, data_model=iter_model, name=self.get_name())

        return dict.keys(self)

    def values(self):
        """
        This method return a list of all the values of the StorageDict.
        Returns:
          list: a list of values
        """

        iter_cols = self._data_model_def.get('cols', None)
        iter_model = {"type": "StorageIter", "name": self.get_name(), "cols": iter_cols}
        if self.storage_id:
            return StorageIter(storage_id=self.storage_id, data_model=iter_model, name=self.get_name())

        return dict.values(self)

    def items(self):
        """
        This method return a list of all the key-value pairs of the StorageDict.
        Returns:
          list: a list of key-value pairs
        """
        if self.storage_id:
            return StorageIter(storage_id=self.storage_id, data_model=self._data_model_def)

        return dict.items(self)

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

    def delete_persistent(self):
        """
        Method to empty all data assigned to a StorageDict.
        """

        log.debug('DELETE PERSISTENT')
        for obj in self.values():
            if isinstance(obj, IStorage):
                obj.delete_persistent()

        storage.StorageAPI.delete_persistent_object(self.storage_id)

        super().delete_persistent()

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
        persistent_result = storage.StorageAPI.get_records(self.storage_id, [key])

        # we need to transform UUIDs belonging to IStorage objects and rebuild them
        final_results = []

        for i, element in enumerate(persistent_result):
            col_type = self.__class__._data_model_def['cols'][i]
            if issubclass(col_type, IStorage):
                # element is not a built-in type
                table_name = self.storage_id + '_' + str(key)
                info = {"name": table_name, "storage_id": element, "class_name": col_type}
                element = build_remotely(info)

            final_results.append(element)

        return final_results

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

        keys = [def_type(val[i]) for i, def_type in enumerate(self._data_model_def['keys'].values())]
        vals = [def_type(val[i]) for i, def_type in enumerate(self._data_model_def['cols'].values())]

        if not self._is_persistent:
            dict.__setitem__(self, keys, vals)
        else:
            storage.StorageAPI.put_records(self.storage_id, [keys], [vals])

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

    def update(self, other=(), **kwargs):
        """
        Updates the current dict with a new dictionary or set of attr,value pairs
        (those must follow the current dict data model).
        Args:
            other: python dictionary or StorageDict. All key,val values in it will
            be inserted in the current dict.
            **kwargs: set of attr:val pairs, to be treated as key,val and inserted
            in the current dict.
        """

        for k, v in other:
            self[k] = v

        for k, v in kwargs.items():
            self[k] = v

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default
