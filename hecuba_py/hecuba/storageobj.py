import numpy as np
from . import log, Parser

from .hnumpy import StorageNumpy
from .IStorage import IStorage, AlreadyPersistentError
from .tools import build_remotely, basic_types, storage_id_from_name, import_class

import storage


class StorageObj(IStorage):
    """
    This class is where information will be stored in Hecuba.
    The information can be in memory, stored in a python dictionary or local variables, or saved in a
    DB(Cassandra), depending on if it's persistent or not.
    """

    def __new__(cls, name='', *args, **kwargs):
        if not cls.DataModelId:
            cls._persistent_props = Parser("ClassField").parse_comments(cls.__doc__)

            def update_type(d):
                import copy
                d_ret = copy.deepcopy(d)
                if d_ret["type"] not in basic_types:
                    d_ret["type"] = import_class("uuid.UUID")
                else:
                    d_ret["type"] = import_class(d_ret["type"])
                return d_ret

            dm = {k: update_type(v) for k, v in cls._persistent_props.items()}
            cls.DataModelId = storage.StorageAPI.add_data_model(dm)

        toret = super(StorageObj, cls).__new__(cls)
        storage_id = kwargs.get('storage_id', None)
        if storage_id is None and name:
            storage_id = storage_id_from_name(name)

        if name or storage_id:
            toret.setID(storage_id)
            toret.set_name(name)
            toret._is_persistent = True
            storage.StorageAPI.register_persistent_object(cls.DataModelId, toret)
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

        storage.StorageAPI.register_persistent_object(self.__class__.DataModelId, self)
        # defined_attrs = [attr for attr in self._persistent_props.keys() if attr in list(set(dir(self)))]

        attrs = []
        values = []
        for obj_name, obj_info in self._persistent_props.items():
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
        for obj_name in self._persistent_props.keys():
            attr = getattr(self, obj_name, None)
            if isinstance(attr, IStorage):
                attr.stop_persistent()

        super().stop_persistent()

    def delete_persistent(self):
        """
            Deletes the Cassandra table where the persistent StorageObj stores data
        """
        log.debug('DELETE PERSISTENT')
        for obj_name in self._persistent_props.keys():
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
            if attribute.startswith('_') or attribute not in self._persistent_props.keys():
                raise ex

        value_info = self._persistent_props[attribute]
        is_istorage_attr = value_info['type'] not in basic_types

        if not self.storage_id:
            if not is_istorage_attr:
                raise AttributeError
            else:
                # We build the object, because Hecuba allows accessing attributes without previous initialization
                info = {"name": '', "storage_id": None}
                info.update(value_info)
                info["built_remotely"] = self._built_remotely
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
        value = attrs[0]

        if is_istorage_attr:
            # If IStorage type, then we rebuild
            attr_name = self.storage_id + '_' + attribute
            # Build the IStorage obj
            info = {"name": attr_name, "storage_id": value}
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
        if attribute[0] is '_' or attribute not in self._persistent_props.keys():
            object.__setattr__(self, attribute, value)
            return

        # Transform numpy.ndarrays and python dicts to StorageNumpy and StorageDicts
        if not isinstance(value, IStorage):
            if isinstance(value, np.ndarray):
                value = StorageNumpy(value)
            elif isinstance(value, dict):
                per_dict = self._persistent_props[attribute]
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
                    storage.StorageAPI.put_records(self.storage_id, [attribute], [attr_id])
                    value.make_persistent(attr_name)
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
        if self.storage_id and item in self._persistent_props.keys():
            storage.StorageAPI.put_records(self.storage_id, self._persistent_props.keys(), [])
        object.__delattr__(self, item)
