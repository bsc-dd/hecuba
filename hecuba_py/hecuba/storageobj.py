import uuid
from collections import namedtuple

import numpy as np
from hecuba import config, log, Parser

from IStorage import IStorage, AlreadyPersistentError, _discrete_token_ranges, _basic_types, _valid_types, \
    _extract_ks_tab
from hnumpy import StorageNumpy
from parser import Parser


class StorageObj(object, IStorage):
    args_names = ["name", "tokens", "storage_id", "istorage_props", "class_name", "built_remotely"]
    args = namedtuple('StorageObjArgs', args_names)
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba' +
                                                  '.istorage (storage_id, class_name, name, tokens,istorage_props) '
                                                  ' VALUES (?,?,?,?,?)')

    """
    This class is where information will be stored in Hecuba.
    The information can be in memory, stored in a python dictionary or local variables, or saved in a
    DB(Cassandra), depending on if it's persistent or not.
    """

    @staticmethod
    def _store_meta(storage_args):
        """
            Saves the information of the object in the istorage table.
            Args:
                storage_args (object): contains all data needed to restore the object from the workers
        """
        log.debug("StorageObj: storing media %s", storage_args)
        try:

            config.session.execute(StorageObj._prepared_store_meta,
                                   [storage_args.storage_id,
                                    storage_args.class_name,
                                    storage_args.name,
                                    storage_args.tokens,
                                    storage_args.istorage_props])
        except Exception as ex:
            log.warn("Error creating the StorageDict metadata: %s, %s", str(storage_args), ex)
            raise ex

    @classmethod
    def _parse_comments(cls, comments):
        parser = Parser("ClassField")
        return parser._parse_comments(comments)

    def __init__(self, name="", tokens=None, storage_id=None, istorage_props=None, built_remotely=False, **kwargs):
        """
            Creates a new storageobj.
            Args:
                name (string): the name of the Cassandra Keyspace + table where information can be found
                tokens (list of tuples): token ranges assigned to the new StorageObj
                storage_id (string):  an unique storageobj identifier
                istorage_props dict(string,string): a map with the storage id of each contained istorage object.
                kwargs: more optional parameters
        """
        log.debug("CREATED StorageObj(%s)", name)
        # Assign private attributes
        self._is_persistent = True if name or storage_id else False
        self._built_remotely = built_remotely
        self._persistent_props = StorageObj._parse_comments(self.__doc__)
        self._persistent_attrs = self._persistent_props.keys()
        self._class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)

        if self._is_persistent:
            if name:
                self._ksp, self._table = _extract_ks_tab(name)
                name = self._ksp + '.' + self._table

            if not storage_id:
                # Rebuild storage id
                storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, name)

            # Retrieve from hecuba istorage the data
            metas = self._get_istorage_attrs(storage_id)

            # If found data, replace the constructor data
            if len(metas) != 0:
                tokens = metas[0].tokens
                istorage_props = metas[0].istorage_props
                name = metas[0].name
                self._ksp, self._table = _extract_ks_tab(name)

        if tokens is None:
            # log.info('using all tokens')
            tokens = [token.value for token in config.cluster.metadata.token_map.ring]
            tokens = _discrete_token_ranges(tokens)

        self._tokens = tokens
        self._storage_id = storage_id
        self._istorage_props = istorage_props

        # Arguments used to build objects remotely
        self._build_args = self.args(name,
                                     self._tokens,
                                     self._storage_id,
                                     self._istorage_props,
                                     self._class_name,
                                     built_remotely)

        if self._is_persistent:
            # If never existed, must create the tables and register
            if not self._built_remotely:
                self._create_tables()
            self._store_meta(self._build_args)

        self._load_attributes()

    def _load_attributes(self):
        """
            Loads the IStorage objects into memory by creating them or retrieving from the backend.
        """
        attrs = []
        for attribute, value_info in self._persistent_props.iteritems():
            if value_info['type'] not in _basic_types:
                # The attribute is an IStorage object
                attrs.append((attribute, getattr(self, attribute)))
        for (attr_name, attr) in attrs:
            setattr(self, attr_name, attr)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.getID() == other.getID()

    def _create_tables(self):
        """
            Setups the python structures used to communicate with the backend.
            Creates the necessary tables on the backend to store the object data.
        """

        log.info("CREATING KEYSPACE AND TABLE %s %s", self._ksp, self._table)

        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (self._ksp, config.replication)
        config.session.execute(query_keyspace)

        query_simple = 'CREATE TABLE IF NOT EXISTS ' + self._ksp + '.' + self._table + \
                       '( storage_id uuid PRIMARY KEY, '
        for key, entry in self._persistent_props.items():
            query_simple += str(key) + ' '
            if entry['type'] != 'dict' and entry['type'] in _valid_types:
                if entry['type'] == 'list' or entry['type'] == 'tuple':
                    query_simple += entry['type'] + '<' + entry['columns'] + '>, '
                else:
                    query_simple += entry['type'] + ', '
            else:
                query_simple += 'uuid, '
        try:
            config.session.execute(query_simple[:-2] + ' )')
        except Exception as ir:
            log.error("Unable to execute %s", query_simple)
            raise ir

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
            raise AlreadyPersistentError("This StorageObj is already persistent [Before:{}.{}][After:{}]",
                                         self._ksp, self._table, name)

        (self._ksp, self._table) = _extract_ks_tab(name)

        if not self._storage_id:
            # Rebuild storage id
            self._storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, self._ksp + '.' + self._table)
            self._build_args = self._build_args._replace(name=self._ksp + '.' + self._table,
                                                         storage_id=self._storage_id)

        # Retrieve from hecuba istorage the data
        metas = self._get_istorage_attrs(self._storage_id)

        # If metadata was found, replace the private attrs
        if len(metas) != 0:
            # Persisted another
            name = metas[0].name
            self._tokens = metas[0].tokens
            self._istorage_props = metas[0].istorage_props
            # Create the interface with the backend to store the object
        self._create_tables()

        self._is_persistent = True
        if self._build_args.storage_id is None:
            self._build_args = self._build_args._replace(name=self._ksp + '.' + self._table,
                                                         storage_id=self._storage_id)
        self._store_meta(self._build_args)

        # Iterate over the objects the user has requested to be persistent
        # retrieve them from memory and make them persistent
        for obj_name, obj_info in self._persistent_props.items():
            try:
                pd = object.__getattribute__(self, obj_name)
                if isinstance(pd, IStorage) and not pd._is_persistent:
                    count = self._count_name_collision(obj_name)
                    sd_name = self._ksp + "." + self._table + "_" + obj_name
                    if count > 1:
                        sd_name += '_' + str(count - 2)
                    pd.make_persistent(sd_name)
                # self is persistent so setting the attribute will store the data and create the appropiate binding
                setattr(self, obj_name, pd)
            except AttributeError:
                # Attribute unset, no action needed
                pass

    def stop_persistent(self):
        """
            The StorageObj stops being persistent, but keeps the information already stored in Cassandra
        """
        for obj_name in self._persistent_attrs:
            attr = getattr(self, obj_name, None)
            if isinstance(attr, IStorage):
                attr.stop_persistent()

        log.debug("STOP PERSISTENT")
        self._is_persistent = False

    def delete_persistent(self):
        """
            Deletes the Cassandra table where the persistent StorageObj stores data
        """
        for obj_name in self._persistent_attrs:
            attr = getattr(self, obj_name, None)
            if isinstance(attr, IStorage):
                attr.delete_persistent()

        query = "TRUNCATE TABLE %s.%s;" % (self._ksp, self._table)
        log.debug("DELETE PERSISTENT: %s", query)
        config.session.execute(query)

        self._is_persistent = False

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
        if attribute.startswith('_') or attribute not in self._persistent_attrs:
            return object.__getattribute__(self, attribute)

        value_info = self._persistent_props[attribute]
        is_istorage_attr = value_info['type'] not in _basic_types
        if not self._is_persistent:
            if not is_istorage_attr:
                return object.__getattribute__(self, attribute)
            else:
                # We are not persistent or the attribute hasn't been assigned an IStorage obj, we build one
                info = {"name": '', "tokens": self._build_args.tokens, "storage_id": None}
                info.update(value_info)
                info["built_remotely"] = self._built_remotely
                value = IStorage.build_remotely(info)
                object.__setattr__(self, attribute, value)
                return value

        '''
        StorageObj is persistent.
        If the attribute is not a built-in object, we might have it in memory. 
        Since python works using references any modification from another reference will affect this attribute,
        which is the expected behaviour. Therefore, is safe to store in-memory the Hecuba objects.
        '''
        try:
            return object.__getattribute__(self, attribute)
        except AttributeError as ex:
            # Not present in memory, we will need to rebuild it
            pass

        query = "SELECT %s FROM %s.%s WHERE storage_id = %s;" % (attribute, self._ksp, self._table, self._storage_id)
        log.debug("GETATTR: %s", query)
        try:
            result = config.session.execute(query)
        except Exception as ex:
            log.warn("GETATTR ex %s", ex)
            raise ex

        try:
            value = result[0][0]
            # if exists but is set to None, the current behaviour is raising AttributeError
            if value is None:
                raise AttributeError('value not found')
        except IndexError as ex:
            if not is_istorage_attr:
                raise AttributeError('value not found')
            value = None

        if is_istorage_attr:
            # If IStorage type, then we rebuild
            count = self._count_name_collision(attribute)
            attr_name = self._ksp + '.' + self._table + '_' + attribute
            if count > 1:
                attr_name += '_' + str(count - 2)
            # Build the IStorage obj
            info = {"name": attr_name, "tokens": self._build_args.tokens, "storage_id": value}
            info.update(value_info)
            info["built_remotely"] = self._built_remotely
            value = IStorage.build_remotely(info)

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
        if attribute[0] is '_' or attribute not in self._persistent_attrs:
            object.__setattr__(self, attribute, value)
            return

        # Transform numpy.ndarrays and python dicts to StorageNumpy and StorageDicts
        if not isinstance(value, IStorage):
            if isinstance(value, np.ndarray):
                value = StorageNumpy(value)
            elif isinstance(value, dict):
                per_dict = self._persistent_props[attribute]
                info = {"name": '', "tokens": self._build_args.tokens, "storage_id": None, "built_remotely": self._built_remotely}
                info.update(per_dict)
                new_value = IStorage.build_remotely(info)
                new_value.update(value)
                value = new_value

        if self._is_persistent:
            # Write attribute to the storage
            if isinstance(value, IStorage):
                if not value._is_persistent:
                    name_collisions = attribute.lower()
                    count = self._count_name_collision(name_collisions)
                    attr_name = self._ksp + '.' + self._table + '_' + name_collisions
                    if count != 0:
                        attr_name += '_' + str(count - 1)
                    value.make_persistent(attr_name)
                # We store the storage_id when the object belongs to an Hecuba class
                values = [self._storage_id, value._storage_id]
                # We store the IStorage object in memory, to avoid rebuilding when it is not necessary
            else:
                values = [self._storage_id, value]

            query = "INSERT INTO %s.%s (storage_id,%s)" % (self._ksp, self._table, attribute)
            query += " VALUES (%s,%s)"

            log.debug("SETATTR: ", query)
            config.session.execute(query, values)

        # We store all the attributes in memory
        object.__setattr__(self, attribute, value)

    def __delattr__(self, item):
        """
        Method that deletes a given attribute from a StorageObj
        Args:
            item: the name of the attribute to be deleted
        """
        if self._is_persistent and item in self._persistent_attrs:
            query = "UPDATE %s.%s SET %s = null WHERE storage_id = %s" % (
                self._ksp, self._table, item, self._storage_id)
            config.session.execute(query)
        object.__delattr__(self, item)
