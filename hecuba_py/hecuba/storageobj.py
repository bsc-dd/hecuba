import numpy as np
from . import log, Parser

from .hnumpy import StorageNumpy
from .IStorage import IStorage, AlreadyPersistentError
from .tools import build_remotely, storage_id_from_name, update_type

import storage


class StorageObj(IStorage):
    """
    This class is where information will be stored in Hecuba.
    The information can be in memory, stored in a python dictionary or local variables, or saved in a
    DB(Cassandra), depending on if it's persistent or not.
    """

    def __new__(cls, name='', *args, **kwargs):
        if not cls._data_model_id:
            persistent_props = Parser("ClassField").parse_comments(cls.__doc__)
            cols = {k: update_type(v['type']) for k, v in persistent_props.items()}
            keys = {}
            cls._data_model_def = {"type": cls.__name__, 'keys': keys, 'cols': cols}
            cls._data_model_id = storage.StorageAPI.add_data_model(cls._data_model_def)

        toret = super(StorageObj, cls).__new__(cls)
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

    def make_persistent(self, name):
        """
            Once a StorageObj has been created, it can be made persistent. This function retrieves the information about
            the Object class schema, and creates a Cassandra table with those parameters, where information will be
            saved from now on, until execution finishes or StorageObj is no longer persistent.
            It also inserts into the new table all information that was in memory assigned to the StorageObj prior to
            this call.
            Args:
                name (string): name with which the table in the DB will be created
        """
        if self._is_persistent:
            raise AlreadyPersistentError("This StorageObj is already persistent {}", name)

        super().make_persistent(name)

        storage.StorageAPI.register_persistent_object(self.__class__._data_model_id, self)
        # defined_attrs = [attr for attr in self._data_model_def.keys() if attr in list(set(dir(self)))]

        attrs = []
        values = []
        for obj_name, obj_type in self._data_model_def["cols"].items():
            try:
                pd = object.__getattribute__(self, obj_name)
                attrs.append(obj_name)
                if isinstance(pd, IStorage):
                    if not pd._is_persistent:
                        sd_name = name + "_" + obj_name
                        pd.make_persistent(sd_name)
                    values.append(pd.getID())
                else:
                    values.append(pd)
            except AttributeError:
                # Attribute unset, no action needed
                pass

        storage.StorageAPI.put_records(self.storage_id, attrs, values)

    def stop_persistent(self):
        """
            The StorageObj stops being persistent, but keeps the information already stored in Cassandra
        """
        log.debug("STOP PERSISTENT")
        for obj_name in self._data_model_def.keys():
            attr = getattr(self, obj_name, None)
            if isinstance(attr, IStorage):
                attr.stop_persistent()

        super().stop_persistent()

    def delete_persistent(self):
        """
            Deletes the Cassandra table where the persistent StorageObj stores data
        """
        log.debug('DELETE PERSISTENT')
        for obj_name in self._data_model_def.keys():
            attr = getattr(self, obj_name, None)
            if isinstance(attr, IStorage):
                attr.delete_persistent()

        storage.StorageAPI.delete_persistent_object(self.storage_id)

        super().delete_persistent()

    def __getattr__(self, attribute):
        """
            Given an attribute, this function returns the value, obtaining it from either:
            a) memory
            b) the Database
            Args:
                attribute: name of the value that we want to obtain
            Returns:
                value: obtained value
        """
        try:
            return object.__getattribute__(self, attribute)
        except AttributeError as ex:
            if attribute.startswith('_') or attribute not in self._data_model_def['cols'].keys():
                raise ex

        value_info = self._data_model_def['cols'][attribute]

        if not self.storage_id:
            if not issubclass(value_info, IStorage):
                raise AttributeError
            else:
                # We build the object, because Hecuba allows accessing attributes without previous initialization
                info = {"name": '', "storage_id": None, 'type': value_info, 'built_remotely': self._built_remotely}
                value = build_remotely(info)
                object.__setattr__(self, attribute, value)
                return value

        '''
        StorageObj is persistent.
        If the attribute is not a built-in object, we might have it in memory. 
        Since python works using references any modification from another reference will affect this attribute,
        which is the expected behaviour. Therefore, is safe to store in-memory the Hecuba objects.
        '''

        attrs = storage.StorageAPI.get_records(self.storage_id, [attribute])

        if not attrs:
            raise AttributeError('Value not found for {}'.format(attribute))
        value = attrs[list(self._data_model_def['cols']).index(attribute)]

        if issubclass(value_info, IStorage):
            # Build the IStorage obj
            info = {"name": self.get_name() + '_' + attribute, "storage_id": value}
            info.update(value_info)
            value = build_remotely(info)

        object.__setattr__(self, attribute, value)
        return value

    def __setattr__(self, attribute, value):
        """
            Given a key and its value, this function saves it (depending on if it's persistent or not):
                a) In memory
                b) In the DB
            Args:
                attribute: name of the value that we want to set
                value: value that we want to save
        """
        if attribute[0] is '_' or attribute not in self._data_model_def.keys():
            object.__setattr__(self, attribute, value)
            return

        # Transform numpy.ndarrays and python dicts to StorageNumpy and StorageDicts
        if not isinstance(value, IStorage):
            if isinstance(value, np.ndarray):
                value = StorageNumpy(value)
            elif isinstance(value, dict):
                per_dict = self._data_model_def[attribute]
                info = {"name": '', "storage_id": None, "built_remotely": self._built_remotely}
                info.update(per_dict)
                new_value = build_remotely(info)
                new_value.update(value)
                value = new_value

        if self.storage_id:
            # Write attribute to the storage
            if isinstance(value, IStorage):
                if not value.storage_id:
                    attr_name = self._name + '_' + attribute
                    attr_id = storage_id_from_name(attr_name)
                    value.make_persistent(attr_name)
                    storage.StorageAPI.put_records(self.storage_id, [attribute], [attr_id])
                else:
                    storage.StorageAPI.put_records(self.storage_id, [attribute], [value.storage_id])
            else:
                storage.StorageAPI.put_records(self.storage_id, [attribute], [value])

        # We store all the attributes in memory
        object.__setattr__(self, attribute, value)

    def __delattr__(self, item):
        """
        Method that deletes a given attribute from a StorageObj
        Args:
            item: the name of the attribute to be deleted
        """
        if self.storage_id and item in self._data_model_def.keys():
            storage.StorageAPI.put_records(self.storage_id, self._data_model_def.keys(), [])
        object.__delattr__(self, item)
