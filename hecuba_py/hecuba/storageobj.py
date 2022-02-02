from collections import namedtuple

import numpy as np
import uuid
from . import config, log, Parser

from .hnumpy import StorageNumpy
from .IStorage import IStorage

from .tools import get_istorage_attrs, build_remotely, storage_id_from_name, basic_types, \
    valid_types, extract_ks_tab


class StorageObj(IStorage):
    args_names = ["name", "columns", "tokens", "storage_id", "class_name", "built_remotely"]
    args = namedtuple('StorageObjArgs', args_names)
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba.istorage'
                                                  '(storage_id, class_name, name, tokens, '
                                                  'columns)'
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
                                    storage_args.columns])
        except Exception as ex:
            log.warn("Error creating the StorageDict metadata: %s, %s", str(storage_args), ex)
            raise ex

    @classmethod
    def _parse_comments(cls, comments):
        parser = Parser("ClassField")
        return parser._parse_comments(comments)


    def _check_schema_and_raise(self, txt):
        """
        Raises an exception if the schema stored in the database does not match
        with the description of the object in memory. This may happen if the
        user specifies an already used name for its data.
        """
        # try to send a useful message if it is a problem with a mismatched schema
        if getattr(self, "_istorage_metas", None) is None:
            self._istorage_metas = get_istorage_attrs(self.storage_id)

        if len(self._columns) != len(self._istorage_metas.columns):
            raise RuntimeError("StorageObj: {}: Metadata does not match specification. Trying {} but stored specification {}".format(txt, self._columns, self._istorage_metas.columns))
        for pos, val in enumerate(self._columns):
           if (self._istorage_metas.columns[pos][0] != val[0]) or (self._istorage_metas.columns[pos][1] != val[1]):
                raise RuntimeError("StorageObj: {}: Metadata does not match specification. Trying {} but stored specification {}".format(txt, self._columns, self._istorage_metas.columns))

    def __init__(self, name=None, storage_id=None, *args, **kwargs):
        """
            Creates a new storageobj.
            Args:
                name (string): the name of the Cassandra Keyspace + table where information can be found
                tokens (list of tuples): token ranges assigned to the new StorageObj
                storage_id (string):  an unique storageobj identifier
                kwargs: more optional parameters
        """

        super().__init__(name=name, storage_id=storage_id, *args, **kwargs)

        self._columns = [] # Empty object
        if getattr(self, "__doc__", None) is not None:
            # Assign private attributes
            self._persistent_props = StorageObj._parse_comments(self.__doc__)
            self._persistent_attrs = self._persistent_props.keys()
            self._columns = [ (k,v['type']) for k,v in self._persistent_props.items()]
        else:
            if not (name or storage_id):
                raise RuntimeError("Volatile StoragObj WITHOUT specification not allowed")

        self._class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)



        self._table = self.__class__.__name__.lower()
        args        = self.args(self._get_name(), self._columns, self._tokens, self.storage_id, self._class_name, self._built_remotely)

        self._build_args = args

        if name or storage_id:  # therefore... are we doing an Instantiation or a Creation? (built_remotely may be used to instantiate a mockup)
            # Field '_istorage_metas' will be set if it exists in HECUBA.istorage
            if getattr(self, "_istorage_metas", None) is None: #Creation
                self._persist_data(name)
            else: #Instantiation
                if getattr(self, "__doc__", None) is not None:
                    # check that the class metadata stored in HECUBA matches the __doc__
                    self._check_schema_and_raise("__init__")
                else: # No documentation passed, used metadata from hecuba.istorage
                    self._columns = self._istorage_metas.columns
                    ksp, table = extract_ks_tab(self._istorage_metas.class_name)
                    self._table = table  #Update table name to match the ClassName from metadata
                    self._persistent_props = {i[0]:{"type":i[1]} for i in self._columns}
                    self._persistent_attrs = self._persistent_props.keys()
                    # Rebuild '_build_args' with modified args
                    self._build_args = self._build_args._replace(columns=self._columns,
                                                                class_name=self._class_name)


        log.debug("CREATED StorageObj(%s)", self._get_name())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.getID() == other.getID()

    def _persist_attributes(self):
        """
        Persist in-memory attributes to the data store
        """
        for attribute in self._persistent_props.keys():
            try:
                val = super().__getattribute__(attribute)
                setattr(self, attribute, val)
            except AttributeError:
                pass

    def _build_is_attribute(self, attribute, persistence_name, storage_id):
        # Build the IStorage obj
        info = {"tokens": self._tokens, "storage_id": storage_id}
        info.update(self._persistent_props[attribute])
        info["built_remotely"] = self._built_remotely
        info['name'] = persistence_name
        return build_remotely(info)

    def _create_tables(self):
        """
            Setups the python structures used to communicate with the backend.
            Creates the necessary tables on the backend to store the object data.
        """

        log.info("CREATING KEYSPACE AND TABLE %s %s",self._ksp, self._table)

        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (self._ksp, config.replication)
        config.executelocked(query_keyspace)

        query_simple = 'CREATE TABLE IF NOT EXISTS ' + self._ksp + '.' + self._table + \
                       '( storage_id uuid PRIMARY KEY, '
        for key, entry in self._persistent_props.items():
            query_simple += str(key) + ' '
            if entry['type'] != 'dict' and entry['type'] in valid_types:
                if entry['type'] == 'list' or entry['type'] == 'tuple':
                    query_simple += entry['type'] + '<' + entry['columns'] + '>, '
                else:
                    query_simple += entry['type'] + ', '
            else:
                query_simple += 'uuid, '
        try:
            config.executelocked(query_simple[:-2] + ' )')
        except Exception as ir:
            log.error("Unable to execute %s", query_simple)
            raise ir

    def sync(self):
        super().sync()

        for attr_name in self._persistent_attrs:
            attr = getattr(super(), attr_name, None)
            if isinstance(attr, IStorage):
                attr.sync()

    def _persist_data(self, name):
        self._table = self.__class__.__name__.lower()

        # Arguments used to build objects remotely
        self._build_args = self.args(self._get_name(),
                                     self._columns,
                                     self._tokens,
                                     self.storage_id,
                                     self._class_name,
                                     self._built_remotely)

        # If never existed, must create the tables and register
        if not self._built_remotely:
            self._create_tables()

        # Iterate over the objects the user has requested to be persistent
        # retrieve them from memory and make them persistent
        self._persist_attributes()

        StorageObj._store_meta(self._build_args)

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
        # Update name
        super().make_persistent(name)
        if getattr(self, "_istorage_metas", None) is not None:
            self._check_schema_and_raise("make_persistent")
        self._persist_data(name)

    def stop_persistent(self):
        """
            The StorageObj stops being persistent, but keeps the information already stored in Cassandra
        """
        log.debug("STOP PERSISTENT")

        for obj_name in self._persistent_attrs:
            try:
                attr = object.__getattribute__(self, obj_name)
            except AttributeError:
                attr = None

            if isinstance(attr, IStorage):
                attr.stop_persistent()

        super().stop_persistent()
        self.storage_id = None

    def delete_persistent(self):
        """
            Deletes the Cassandra table where the persistent StorageObj stores data
        """
        log.debug("DELETE PERSISTENT: %s", self._table)

        for obj_name in self._persistent_attrs:
            attr = getattr(self, obj_name, None)
            if isinstance(attr, IStorage):
                attr.delete_persistent()

        # TODO Drop table _ksp._table if it just contains a single element (non-perfomant :(
        query = "DELETE FROM {}.{} where storage_id={}".format(self._ksp, self._table, self.storage_id)
        config.session.execute(query)

        query = "DELETE FROM hecuba.istorage where storage_id={}".format(self.storage_id)
        config.session.execute(query)

        super().delete_persistent()
        self.storage_id = None

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
            return super().__getattribute__(attribute)

        is_istorage_attr = self._persistent_props[attribute]["type"] not in basic_types

        if not self.storage_id:
            if is_istorage_attr:
                value = self._build_is_attribute(attribute, persistence_name=None, storage_id=None)
                super().__setattr__(attribute, value)
            return super().__getattribute__(attribute)

        '''
        StorageObj is persistent.
        If the attribute is not a built-in object, we might have it in memory. 
        Since python works using references any modification from another reference will affect this attribute,
        which is the expected behaviour. Therefore, is safe to store in-memory the Hecuba objects.
        '''
        try:
            return super().__getattribute__(attribute)
        except AttributeError as ex:
            # Not present in memory, we will need to rebuild it
            pass

        query = "SELECT %s FROM %s.%s WHERE storage_id = %s;" % (attribute, self._ksp, self._table, self.storage_id)
        log.debug("GETATTR: %s", query)
        try:
            result = config.session.execute(query)
        except Exception as ex:
            log.warn("GETATTR ex %s", ex)
            raise ex

        try:
            value = result[0][0]
            # if exists but is set to None, the current behaviour is raising AttributeError
            if not is_istorage_attr and value is None:
                raise AttributeError('value not found')
        except IndexError as ex:
            if not is_istorage_attr:
                raise AttributeError('value not found')
            value = None
        except TypeError as ex:
            log.warn("ERROR ON QUERY RESULT {}".format(str(result)))
            raise ex

        if is_istorage_attr:
            # Value is uuid or None, because it was not found

            attr_name = None
            if value is None:
                # Value not found, persist it BY NAME using a random name so we can retrieve it later
                attr_name = attribute.lower()
                my_name   = self._get_name()
                trailing_name = my_name[my_name.rfind('.') + 1:]
                number    = uuid.uuid4() # Random value
                attr_name = self._ksp + "." + ("O" + str(number).replace('-','_') + trailing_name + attr_name)[:40]
                value = self._build_is_attribute(attribute, persistence_name=attr_name, storage_id=None)
                # Following lines emulate "self.__setattr__(attribute, value)" without the checks
                values = [self.storage_id, value.storage_id]
                query = "INSERT INTO %s.%s (storage_id,%s)" % (self._ksp, self._table, attribute)
                query += " VALUES (%s,%s)"
                log.debug("SETATTR: " + query)
                config.session.execute(query, values)

            else :
                value = self._build_is_attribute(attribute, persistence_name=attr_name, storage_id=value)

        super().__setattr__(attribute, value)
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
        if attribute[0] == '_' or attribute not in getattr(self, "_persistent_attrs", []):
            super().__setattr__(attribute, value)
            return

        # Transform numpy.ndarrays and python dicts to StorageNumpy and StorageDicts
        if not isinstance(value, IStorage):
            if isinstance(value, np.ndarray):
                value = StorageNumpy(value)
            elif isinstance(value, dict):
                per_dict = self._persistent_props[attribute]
                info = {"name": '', "tokens": self._build_args.tokens, "storage_id": None,
                        "built_remotely": self._built_remotely}
                info.update(per_dict)
                new_value = build_remotely(info)
                new_value.update(value)
                value = new_value

        if self.storage_id:
            # Write attribute to the storage
            if isinstance(value, IStorage):
                if not value.storage_id:
                    # Value is volatile, persist it BY NAME using a random name so we can retrieve it later
                    attr_name = attribute.lower()
                    my_name = self._get_name()
                    trailing_name = my_name[my_name.rfind('.') + 1:]
                    number = uuid.uuid4() # Random value
                    name   = self._ksp + "." + ("O" + str(number).replace('-','_') + trailing_name + attr_name)[:40]
                    value.make_persistent(name)   # Persist BY NAME
                # We store the storage_id when the object belongs to an Hecuba class
                values = [self.storage_id, value.storage_id]
                # We store the IStorage object in memory, to avoid rebuilding when it is not necessary
            else:
                values = [self.storage_id, value]

            query = "INSERT INTO %s.%s (storage_id,%s)" % (self._ksp, self._table, attribute)
            query += " VALUES (%s,%s)"

            log.debug("SETATTR: " + query)
            config.session.execute(query, values)

        # We store all the attributes in memory
        super().__setattr__(attribute, value)

    def __delattr__(self, name):
        """
        Method that deletes a given attribute from a StorageObj
        Args:
            item: the name of the attribute to be deleted
        """
        super().__delattr__(name)

        if self.storage_id and name in self._persistent_attrs:
            query = "UPDATE %s.%s SET %s = null WHERE storage_id = %s" % (
                self._ksp, self._table, name, self.storage_id)
            config.session.execute(query)

    def sync(self):
        """
        Wait until all pending stores to Cassandra have been finished.
        """
        if not self.storage_id:
            return

        # Persistent Object
        for attribute in self._persistent_props.keys():
            try:
                val = super().__getattribute__(attribute)
                if isinstance(val, IStorage):
                    log.debug("StorageObj sync: %s.%s of type %s", self.storage_id, attribute,type(val))
                    val.sync()
            except AttributeError as ex:
                # Not present in memory
                pass

    def split(self):
        raise NotImplementedError("Split is not supported on StorageObjects");

