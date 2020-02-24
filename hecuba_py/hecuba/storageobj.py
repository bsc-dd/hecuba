import uuid

import numpy as np

import storage
from .IStorage import IStorage, AlreadyPersistentError
from .tools import storage_id_from_name, transform_to_dm


class StorageObj(IStorage):
    """
    This class is where information will be stored in Hecuba.
    The information can be in memory, stored in a python dictionary or local variables, or saved in a
    DB(Cassandra), depending on if it's persistent or not.
    """

    def __new__(cls, name='', *args, **kwargs):
        if not cls._data_model_id:
            try:
                cls._data_model_def = kwargs['data_model']
            except KeyError:
                cls._data_model_def = dict()
                cls._data_model_def['type'] = cls
                cls._data_model_def['value_id'] = {'k': uuid.UUID}
                cls._data_model_def['fields'] = transform_to_dm(cls)
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

        value_dict = {}
        for obj_name, obj_type in self._data_model_def["fields"].items():
            try:
                pd = object.__getattribute__(self, obj_name)
            except AttributeError:
                # Attribute unset, no action needed
                continue
            value_dict[obj_name] = pd

        storage.StorageAPI.put_record(self.storage_id, {'k': self.storage_id}, value_dict)

    def stop_persistent(self):
        """
            The StorageObj stops being persistent, but keeps the information already stored in Cassandra
        """

        super().stop_persistent()

    def delete_persistent(self):
        """
            Deletes the Cassandra table where the persistent StorageObj stores data
        """
        response = storage.StorageAPI.delete_persistent_object(self.storage_id)
        super().delete_persistent()
        return response

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
            if attribute.startswith('_') or attribute not in self._data_model_def['fields'].keys():
                raise ex

        value_info = self._data_model_def['fields'][attribute]

        if not self.storage_id:
            if not issubclass(value_info, IStorage):
                raise AttributeError
            else:
                # We build the object, because Hecuba allows accessing attributes without previous initialization
                value = value_info(data_model=self._data_model_def["fields"][attribute], build_remotely=True)
                object.__setattr__(self, attribute, value)
                return value

        assert self._is_persistent

        attr = storage.StorageAPI.get_record(self.storage_id, {
            'k': self.storage_id})

        # if issubclass(value_info, IStorage):
        #     # Build the IStorage obj
        #     attr = value_info(name=self.get_name() + '_' + attribute, storage_id=attr,
        #                       data_model=self._data_model_def["fields"][attribute], build_remotely=True)
        # elif not attr:
        #     raise AttributeError('Value not found for {}'.format(attribute))
        index = list(self._data_model_def["fields"]).index(attribute)
        object.__setattr__(self, attribute, attr[index])
        return attr[index]

    def __setattr__(self, attribute, value):
        """
            Given a key and its value, this function saves it (depending on if it's persistent or not):
                a) In memory
                b) In the DB
            Args:
                attribute: name of the value that we want to set
                value: value that we want to save
        """
        if attribute[0] is '_' or attribute not in self._data_model_def["fields"].keys():
            object.__setattr__(self, attribute, value)
            return

        # Transform numpy.ndarrays and python dicts to StorageNumpy and StorageDicts
        if not isinstance(value, IStorage):
            if isinstance(value, np.ndarray):
                pass
                # value = StorageNumpy(value)
            elif isinstance(value, dict):
                obj_class = self._data_model_defDataModelId["fields"][attribute]["type"]
                value = obj_class(data_model=self._data_model_def["fields"][attribute], build_remotely=False)

        if self.storage_id:
            # Write attribute to the storage
            if isinstance(value, IStorage):
                storage.StorageAPI.put_record(self.storage_id, {'k': self.storage_id}, {attribute: value.storage_id})
            else:
                storage.StorageAPI.put_record(self.storage_id, {'k': self.storage_id}, {attribute: value})

        # We store all the attributes in memory
        object.__setattr__(self, attribute, value)

    def __delattr__(self, item):
        """
        Method that deletes a given attribute from a StorageObj
        Args:
            item: the name of the attribute to be deleted
        """
        if self.storage_id and item in self._data_model_def["fields"].keys():
            storage.StorageAPI.put_record(self.storage_id, {'k': self.storage_id}, {item: None})
        object.__delattr__(self, item)
