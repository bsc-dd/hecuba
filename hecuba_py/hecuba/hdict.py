import uuid
from collections import Iterable, defaultdict
from collections import Mapping
from collections import namedtuple

from cassandra import OperationTimedOut

import numpy as np
from . import config, log, Parser
from .storageiter import NamedItemsIterator, NamedIterator
from .hnumpy import StorageNumpy
from hecuba.hfetch import Hcache

from .IStorage import IStorage
from .tools import get_istorage_attrs, build_remotely, basic_types, _min_token, _max_token, storage_id_from_name


class EmbeddedSet(set):
    '''
    father is the dictionary containing the set
    keys are the keys names of the set in the dictionary
    values is the initializing set
    '''

    def __init__(self, father, keys, values=None):
        super(EmbeddedSet, self).__init__()
        self._father = father
        self._keys = keys
        if values is not None:
            if len(self) != 0:
                self.clear()
            if isinstance(values, set):
                for value in values:
                    self.add(value)
            else:
                raise Exception("Set expected.")

    def add(self, value):
        keys = self._keys[:]
        if not isinstance(value, Iterable) or isinstance(value, str):
            keys.append(value)
        else:
            keys += list(value)
        return self._father.__setitem__(keys, [])

    def remove(self, value):
        if value in self:
            keys = self._keys[:]
            if not isinstance(value, Iterable) or isinstance(value, str):
                keys.append(value)
            else:
                keys += list(value)
            return self._father.__delitem__(keys)
        else:
            raise KeyError

    def discard(self, value):
        try:
            if value in self:
                keys = self._keys[:]
                if not isinstance(value, Iterable) or isinstance(value, str):
                    keys.append(value)
                else:
                    keys += list(value)
                return self._father.__delitem__(keys)
        except KeyError as ex:
            pass

    def __len__(self):
        query = "SELECT COUNT(*) FROM %s.%s WHERE " % (self._father._ksp, self._father._table)
        query = ''.join([query, self._join_keys_query()])

        try:
            result = config.session.execute(query)
            return result[0][0]
        except Exception as ir:
            log.error("Unable to execute %s", query)
            raise ir

    def __contains__(self, value):
        keys = self._keys[:]
        if not isinstance(value, Iterable) or isinstance(value, str):
            keys.append(value)
        else:
            keys += list(value)
        return self._father.__contains__(keys)

    def __iter__(self):
        keys_set = ""
        for key in self._father._get_set_types():
            keys_set += key[0] + ", "
        query = "SELECT %s FROM %s.%s WHERE " % (keys_set[:-2], self._father._ksp, self._father._table)
        query = ''.join([query, self._join_keys_query()])

        try:
            result = config.session.execute(query)
            if len(self._father._get_set_types()) == 1:
                result = map(lambda x: x[0], result)
            else:
                result = map(lambda x: tuple(x), result)
            return iter(result)
        except Exception as ir:
            log.error("Unable to execute %s", query)
            raise ir

    def _join_keys_query(self):
        keys = []
        for pkey, key in zip(self._father._primary_keys, self._keys):
            if pkey["type"] == "text":
                actual_key = "'%s'" % key
            else:
                actual_key = "%s" % key
            keys.append(" = ".join([pkey["name"], actual_key]))
        all_keys = " and ".join(keys)

        return all_keys

    def union(self, *others):
        result = set()
        for value in self:
            result.add(value)
        for other in others:
            for value in other:
                result.add(value)
        return result

    def intersection(self, *others):
        result = set()
        for value in self:
            in_all_others = True
            for other in others:
                try:
                    if value not in other:
                        in_all_others = False
                        break
                except KeyError:
                    in_all_others = False
                    break
            if in_all_others:
                result.add(value)
        return result

    def difference(self, *others):
        result = set()
        for value in self:
            in_any_other = False
            for other in others:
                try:
                    if value in other:
                        in_any_other = True
                        break
                except KeyError:
                    pass
            if not in_any_other:
                result.add(value)
        return result

    def update(self, *others):
        for other in others:
            for value in other:
                self.add(value)
        return self

    def issubset(self, other):
        if len(self) > len(other):
            return False
        for value in self:
            if value not in other:
                return False
        return True

    def issuperset(self, other):
        if len(self) < len(other):
            return False
        for value in other:
            if value not in self:
                return False
        return True

    def __eq__(self, other):
        return self._father.__eq__(other._father) and self._keys == other._keys

    def __ne__(self, other):
        return not (self.__eq__(other))

    def __lt__(self, other):
        return self.__ne__(other) and self.issubset(other)

    def __le__(self, other):
        return self.issubset(other)

    def __gt__(self, other):
        return self.__ne__(other) and self.issuperset(other)

    def __ge__(self, other):
        return self.issuperset(other)

    def clear(self):
        for value in self._father[tuple(self._keys)]:
            self.remove(value)


class StorageDict(IStorage, dict):
    # """
    # Object used to access data from workers.
    # """

    args_names = ["name", "primary_keys", "columns", "tokens", "storage_id", "indexed_on", "class_name",
                  "built_remotely"]
    args = namedtuple('StorageDictArgs', args_names)
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba.istorage'
                                                  '(storage_id, class_name, name, tokens, '
                                                  'primary_keys, columns, indexed_on)'
                                                  'VALUES (?,?,?,?,?,?,?)')

    @staticmethod
    def _store_meta(storage_args):
        """
        Method to update the info about the StorageDict in the DB metadata table
        Args:
            storage_args: structure with all data needed to update the metadata
        """
        log.debug("StorageDict: storing metas %s", storage_args)

        try:
            config.session.execute(StorageDict._prepared_store_meta,
                                   [storage_args.storage_id, storage_args.class_name,
                                    storage_args.name,
                                    storage_args.tokens, storage_args.primary_keys,
                                    storage_args.columns, storage_args.indexed_on])
        except Exception as ex:
            log.error("Error creating the StorageDict metadata: %s %s", storage_args, ex)
            raise ex



    def _initialize_stream_capability(self, topic_name=None):
        super()._initialize_stream_capability(topic_name)
        if topic_name is not None:
            self._hcache.enable_stream(self._topic_name, {'kafka_names': str.join(",",config.kafka_names)})

    def __init__(self, name=None, primary_keys=None, columns=None, indexed_on=None, storage_id=None, **kwargs):
        """
        Creates a new StorageDict.

        Args:
            name (string): the name of the collection/table (keyspace is optional)
            primary_keys (list(tuple)): a list of (key,type) primary keys (primary + clustering).
            columns (list(tuple)): a list of (key,type) columns
            tokens (list): list of tokens
            storage_id (string): the storage id identifier
            indexed_on (list): values that will be used as index
            kwargs: other parameters
        """

        super(StorageDict, self).__init__(name=name, storage_id=storage_id, **kwargs)
        log.debug("CREATE StorageDict(%s,%s)", primary_keys, columns)

        '''
        yolandab
        kwargs of the init should contain metas: all the row in the istorage if exists
        after super().__init__
                    if kwargs is empty --> this is a new object
                        generate build args parsing the _doc_ string or using the parameters
                        we need to generate the column info of sets with the format to persist it (name--> _set_)
                        if name or storage id --> call to store_metas
                    else --> this is an already existing objects
                        metas and tokens should form the attributes of self
                        we need to convert the column info of sets to the format in memory ( _set_name --> name)
            TODO: implement a cleaner version of embedded sets
        '''
        build_column = None
        build_keys = None
        if self.__doc__ is not None:
            self._persistent_props = self._parse_comments(self.__doc__)
            self._primary_keys = self._persistent_props['primary_keys']
            self._columns = self._persistent_props['columns']
            self._indexed_on = self._persistent_props.get('indexed_on', indexed_on)

        # Field '_istorage_metas' will be set if it exists in HECUBA.istorage
        initialized = (getattr(self, '_istorage_metas', None) is not None)
        if not initialized and self.__doc__ is None:
            #info is not in the doc string, should be passed in the parameters
            if primary_keys == None or columns == None:
                raise RuntimeError ("StorageDict: missed specification. Type of Primary Key or Column undefined")
            self._primary_keys = primary_keys
            self._columns = columns
            self._indexed_on = indexed_on

        if initialized: #object already in istorage

            # if (primary_keys is not None or columns is not None):
            #    raise RuntimeError("StorageDict: Trying to define a new schema, but it is already persistent")
            #    --> this check would be necessary if passing columns/key spec
            #    as parameter was part of the user interface. As it is intended
            #    just for internal use we skip this check. If the spec does not
            #    match the actual schema access to the object will fail.

            if getattr(self, "_persistent_props", None) is not None: # __doc__ and disk: do they match?
                self._check_schema_and_raise("__init__")

            else: # _persistent_props == None (only in disk)
                # Parse _istorage_metas to fulfill the _primary_keys, _columns
                self._primary_keys = self._istorage_metas.primary_keys
                self._columns = self._istorage_metas.columns
                build_column = self._columns # Keep a copy from the disk to avoid recalculate it later
                build_keys = self._primary_keys # Keep a copy from the disk to avoid recalculate it later
                self._indexed_on = self._istorage_metas.indexed_on
                #we manipulate the info about sets retrieved from istorage
                # (_set_s1_0,int), (_set_s1_1,int) --> {name: s1, type: set , column:((s1_0, int), (s1_1, int))}
                has_embedded_set = False
                set_pks = []
                normal_columns = []
                for column_name, column_type in self._columns:
                    if column_name.find("_set_") == 0:
                        attr_name=column_name[5:] # Remove '_set_' The attribute name also contains the "column_name" needed later...
                        set_pks.append((attr_name, column_type))
                        has_embedded_set = True
                    else:
                        normal_columns.append((column_name, column_type))
                if has_embedded_set: # Embedded set has a different layout {name,type:set, columns:[(name,type),(name,type)]}
                    column_name = attr_name.split("_",1)[0] # Get the 1st name (attr_1, attr_2... -> attr or attr -> attr)
                    self._columns = [{"name": column_name, "type": "set", "columns": set_pks}]
                else:
                    self._columns = [{"type": col[1], "name": col[0]} for col in normal_columns]


        # COMMON CODE: new and instantiation
        # Special case:Do we have an embedded set?
        self._has_embedded_set = False
        if isinstance(self._columns[0], dict):
            if self._columns[0]['type'] == 'set':
                self._has_embedded_set = True

        self._primary_keys = [{"type": key[1], "name": key[0]} if isinstance(key, tuple) else key
                                for key in self._primary_keys]
        self._columns = [{"type": col[1], "name": col[0]} if isinstance(col, tuple) else col
                                for col in self._columns]
        # POST: _primary_keys and _columns are list of DICTS> [ {name:..., type:...}, {name:..., type:set, columns:[(name,type),...]},...]
        log.debug("CREATED StorageDict(%s,%s)", self._primary_keys, self._columns)
        key_names = [key["name"] for key in self._primary_keys]
        column_names = [col["name"] for col in self._columns]
        self._key_column_builder = namedtuple('row', key_names + column_names)

        if len(key_names) > 1:
            self._key_builder = namedtuple('row', key_names)
        else: # 1
            self._key_builder = None

        if self._has_embedded_set:
            set_names = [colname for (colname, dt) in self._get_set_types()]
            self._column_builder = namedtuple('row', set_names)
        elif len(column_names) > 1:
            self._column_builder = namedtuple('row', column_names)
        else:
            self._column_builder = None

        self._k_size = len(key_names)

        class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)

        if build_keys == None:
            build_keys = [(key["name"], key["type"]) for key in self._primary_keys]

        # Define 'build_column': it will contain the column info stored in istorage. For the sets we manipulate the parsed data
        if build_column == None:
            build_column = []
            for col in self._columns:
                if col["type"] == "set":
                    types = col["columns"]
                    for t in types:
                        build_column.append(("_set_" + t[0], t[1]))
                else:
                    build_column.append((col["name"], col["type"]))


        self._build_args = self.args(self._get_name(), build_keys, build_column, self._tokens,
                                     self.storage_id, self._indexed_on, class_name, self._built_remotely)

        if name and storage_id and (storage_id != storage_id_from_name(name)): # instantiating an splitted object
            self._persist_metadata()
        elif name or storage_id: # instantiating a persistent object
            if initialized: # already existint
                self._setup_hcache()
            else: # new object
                self._persist_metadata()

        if self._is_stream():
            log.debug("StorageDict with streaming capability")
            if self.storage_id is not None:
                topic_name=str(self.storage_id)
            else:
                topic_name=None
            self._initialize_stream_capability(topic_name)

    @classmethod
    def _parse_comments(self, comments):
        parser = Parser("TypeSpec")
        return parser._parse_comments(comments)

    def __contains__(self, key):
        """
        Method that checks if a given key exists in a StorageDict.
        Args:
            key: the position that we want to check if exists.
        Returns:
            boolean (true - exists, false - doesn't exist).
        """
        if not self.storage_id:
            return dict.__contains__(self, key)
        else:
            try:
                # TODO we should save this value in a cache
                self._hcache.get_row(self._make_key(key))
                return True
            except Exception as ex:
                log.warn("persistentDict.__contains__ ex %s", ex)
                return False

    def _create_tables(self):
        # Prepare data
        persistent_keys = [(key["name"], "tuple<" + ",".join(key["columns"]) + ">") if key["type"] == "tuple"
                           else (key["name"], key["type"]) for key in self._primary_keys] + self._get_set_types()
        persistent_values = []
        if not self._has_embedded_set:
            for col in self._columns:
                if col["type"] == "tuple":
                    persistent_values.append({"name": col["name"], "type": "tuple<" + ",".join(col["columns"]) + ">"})
                elif col["type"] not in basic_types:
                    persistent_values.append({"name": col["name"], "type": "uuid"})
                else:
                    persistent_values.append({"name": col["name"], "type": col["type"]})

        key_names = [col[0] if isinstance(col, tuple) else col["name"] for col in persistent_keys]

        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (self._ksp, config.replication)
        try:
            log.debug('MAKE PERSISTENCE: %s', query_keyspace)
            config.executelocked(query_keyspace)
        except Exception as ex:
            log.warn("Error creating the StorageDict keyspace %s, %s", (query_keyspace), ex)
            raise ex

        persistent_columns = [(col["name"], col["type"]) for col in persistent_values]

        query_table = "CREATE TABLE IF NOT EXISTS %s.%s (%s, PRIMARY KEY (%s));" \
                      % (self._ksp,
                         self._table,
                         ",".join("%s %s" % tup for tup in persistent_keys + persistent_columns),
                         str.join(',', key_names))
        try:
            log.debug('MAKE PERSISTENCE: %s', query_table)
            config.executelocked(query_table)
        except Exception as ex:
            log.warn("Error creating the StorageDict table: %s %s", query_table, ex)
            raise ex

        if hasattr(self, '_indexed_on') and self._indexed_on is not None:
            index_query = 'CREATE CUSTOM INDEX IF NOT EXISTS ' + self._table + '_idx ON '
            index_query += self._ksp + '.' + self._table + ' (' + str.join(',', self._indexed_on) + ') '
            index_query += "using 'es.bsc.qbeast.index.QbeastIndex';"
            try:
                config.executelocked(index_query)
            except Exception as ex:
                log.error("Error creating the Qbeast custom index: %s %s", index_query, ex)
                raise ex
            trigger_query = "CREATE TRIGGER IF NOT EXISTS %s%s_qtr ON %s.%s USING 'es.bsc.qbeast.index.QbeastTrigger';" % \
                            (self._ksp, self._table, self._ksp, self._table)
            try:
                config.executelocked(trigger_query)
            except Exception as ex:
                log.error("Error creating the Qbeast trigger: %s %s", trigger_query, ex)
                raise ex

    def _persist_data_from_memory(self):
        for k, v in super().items():
            self[k] = v
        if config.max_cache_size != 0: #if C++ cache is enabled, clear Python memory, otherwise keep it
            super().clear()

    def sync(self):
        super().sync()
        self._hcache.flush()

    def _setup_hcache(self):
        key_names = [key["name"] for key in self._primary_keys]
        key_names = key_names + [name for name, dt in self._get_set_types()]

        persistent_values = []
        if not self._has_embedded_set:
            persistent_values = [{"name": col["name"]} for col in self._columns]
        if self._tokens is None:
            raise RuntimeError("Tokens for object {} are null".format(self._get_name()))
        self._hcache_params = (self._ksp, self._table,
                               self.storage_id,
                               self._tokens, key_names, persistent_values,
                               {'cache_size': config.max_cache_size,
                                'writer_par': config.write_callbacks_number,
                                'writer_buffer': config.write_buffer_size,
                                'timestamped_writes': config.timestamped_writes})
        log.debug("HCACHE params %s", self._hcache_params)
        self._hcache = Hcache(*self._hcache_params)

    def _make_key(self, key):
        """
        Method used to pass the key data to the StorageDict cache in a proper way.
        Args:
            key: the data that needs to get the correct format
        """
        if isinstance(key, str) or not isinstance(key, Iterable):
            if len(self._primary_keys) == 1:
                return [key]
            else:
                raise Exception('missing a primary key')

        if isinstance(key, Iterable) and len(key) == len(self._primary_keys):
            return list(key)
        elif self._has_embedded_set and isinstance(key, Iterable) and len(key) == (
                len(self._primary_keys) + len(self._get_set_types())):
            return list(key)
        else:
            raise Exception('wrong primary key {}'.format(type(key)))

    @staticmethod
    def _make_value(value):
        """
        Method used to pass the value data to the StorageDict cache in a proper way.
        Args:
            value: the data that needs to get the correct format
        """
        if issubclass(value.__class__, IStorage):
            return [value.storage_id]
        elif isinstance(value, str) or not isinstance(value, Iterable) or isinstance(value, np.ndarray):
            return [value]
        elif isinstance(value, tuple):
            return [value]
        elif isinstance(value, Iterable):
            val = []
            for v in value:
                if isinstance(v, IStorage):
                    val.append(v.storage_id)
                else:
                    val.append(v)
            return val
        else:
            return list(value)

    def _count_elements(self, query):
        try:
            result = config.session.execute(query)
            return result[0][0]
        except OperationTimedOut as ex:
            import warnings
            warnings.warn("len() operation on {} from class {} failed by timeout."
                          "Use len() on split() results if you must".format(self._get_name(), self.__class__.__name__))
            raise ex
        except Exception as ir:
            log.error("Unable to execute %s", query)
            raise ir

    def __iter__(self):
        """
        Method that overloads the python dict basic iteration, which returns
        an iterator over the dictionary keys.
        """
        return self.keys()

    def _persist_metadata(self):
        """
        Private Method to create tables, setup the cache and store the metadata
        of a StorageDict.
        Used for NEW storage dicts, that do no need to persist any data.
        """
        if not self._built_remotely:
            self._create_tables()
        self._setup_hcache()
        StorageDict._store_meta(self._build_args)

    def _persist_data(self, name):
        """
        Private Method to store a StorageDict into cassandra
        This will make it use a persistent DB as the main location
        of its data.
        Args:
            name:
        """
        # Update local StorageDict metadata
        self._build_args = self._build_args._replace(storage_id=self.storage_id, name=self._ksp + "." + self._table,
                                                     tokens=self._tokens)
        self._persist_metadata()
        self._persist_data_from_memory()

    def make_persistent(self, name):
        """
        Method to transform a StorageDict into a persistent object.
        This will make it use a persistent DB as the main location
        of its data.
        Args:
            name:
        """
        super().make_persistent(name)
        if getattr(self, "_istorage_metas", None) is not None:
            self._check_schema_and_raise("make_persistent")
        self._persist_data(name)

    def stop_persistent(self):
        """
        Method to turn a StorageDict into non-persistent.
        """
        super().stop_persistent()
        log.debug('STOP PERSISTENCE: %s', self._table)
        self._hcache = None
        self.storage_id = None

    def delete_persistent(self):
        """
        Method to empty all data assigned to a StorageDict.
        """
        self.sync()
        super().delete_persistent()
        log.debug('DELETE PERSISTENT: %s', self._table)
        query = "DROP TABLE %s.%s;" % (self._ksp, self._table)
        config.session.execute(query)

        query = "DELETE FROM hecuba.istorage where storage_id={}".format(self.storage_id)
        config.session.execute(query)
        self.storage_id = None

    def __delitem__(self, key):
        """
        Method to delete a specific entry in the dict in the key position.
        Args:
            key: position of the entry that we want to delete
        """
        if not self.storage_id:
            dict.__delitem__(self, key)
        elif self._has_embedded_set:
            self._hcache.delete_row(key)
        elif isinstance(key, Iterable) and not isinstance(key, str):
            self._hcache.delete_row(list(key))
        else:
            self._hcache.delete_row([key])

    def __create_embeddedset(self, key, val=None):
        if not isinstance(key, Iterable) or isinstance(key, str):
            return EmbeddedSet(self, [key], val)
        else:
            return EmbeddedSet(self, list(key), val)

    def _check_schema_and_raise(self, txt):
        """
        Raises an exception if the schema stored in the database does not match
        with the description of the object in memory. This may happen if the
        user specifies an already used name for its data.
        PRE:
            self._istorage_metas contains a list of tuples (name, type)
            self._primary_keys contains a list of tuples (name, type) or list of dicts {'name':value, 'type':value}
            self._columns may contain:
                        a list of tuples (name, type) or
                        a list of dicts {'name':value, 'type':value}  or
                        a list of dicts with a set {'name':value, 'type':'set','columns':[(name1,type1),....]}
        """
        # TODO: Change parser to have a consistent behaviour
        # try to send a useful message if it is a problem with a mismatched schema
        if getattr(self, "_istorage_metas", None) is None:
            self._istorage_metas = get_istorage_attrs(self.storage_id)

        if len(self._primary_keys) != len(self._istorage_metas.primary_keys):
            raise RuntimeError("StorageDict: {}: key Metadata does not match specification. Trying {} but stored specification {}".format(txt, self._primary_keys, self._istorage_metas.primary_keys))
        pk = [{"type": key[1], "name": key[0]} if isinstance(key, tuple) else key
                                for key in self._primary_keys]

        for pos, key in enumerate(pk):
            if self._istorage_metas.primary_keys[pos][0] != key['name'] or self._istorage_metas.primary_keys[pos][1] != key['type']:
                raise RuntimeError("StorageDict: {}: key Metadata does not match specification. Trying {} but stored specification {}".format(txt, self._primary_keys, self._istorage_metas.primary_keys))


        columns = self._columns
        # Treat the embedded set case...
        if type(self._columns[0]) == dict:
            if self._columns[0]['type'] == 'set':
                columns = self._columns[0]['columns']
        if len(columns) != len(self._istorage_metas.columns):
            raise RuntimeError("StorageDict: {}: column Metadata does not match specification. Trying {} but stored specification {}".format(txt, self._columns, self._istorage_metas.columns))
        columns = [{"type": col[1], "name": col[0]} if isinstance(col, tuple) else col
                                for col in columns]
        for pos, val in enumerate(columns):
            #istorage_metas.columns[pos] -->[(_set_s1_0,int),(_set_s1_1,int)]
            mykey = self._istorage_metas.columns[pos][0]
            mytype= self._istorage_metas.columns[pos][1]

            if mykey.find("_set_") == 0:
                mykey = mykey[5:] # Skip the '_set_' '_set_s1_0' ==> 's1_0' TODO Change the set identification method
            if (mykey != val['name']) or (mytype != val['type']):
                raise RuntimeError("StorageDict: {}: column Metadata does not match specification. Trying {} but stored specification {}".format(txt, self._columns, self._istorage_metas.columns))

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
        elif self._has_embedded_set:
            return self.__create_embeddedset(key=key)
        else:
            # Returns always a list with a single entry for the key
            if config.max_cache_size == 0: # if C++ cache is disabled, use Python memory
                try:
                    result = dict.__getitem__(self, key)
                    return result
                except:
                    pass

            persistent_result = self._hcache.get_row(self._make_key(key))

            log.debug("GET ITEM %s[%s]", persistent_result, persistent_result.__class__)

            # we need to transform UUIDs belonging to IStorage objects and rebuild them
            # TODO hcache should return objects of the class uuid, not str
            final_results = []
            for index, col in enumerate(self._columns):
                col_type = col["type"]
                element = persistent_result[index]
                if col_type not in basic_types:
                    # element is not a built-in type
                    info = {"storage_id": element, "tokens": self._build_args.tokens, "class_name": col_type}
                    element = build_remotely(info)

                final_results.append(element)

            if self._column_builder is not None:
                return self._column_builder(*final_results)
            else:
                return final_results[0]

    def __make_val_persistent(self, val, col=0):
        if isinstance(val, list):
            for index, element in enumerate(val):
                val[index] = self.__make_val_persistent(element, index)
        elif isinstance(val, IStorage) and not val._is_persistent:
            valstorage_id = uuid.uuid4()
            attribute = self._columns[col]["name"]

            name = self._ksp + "." + ("D" + str(valstorage_id).replace('-','_') + self._table + attribute)[:40] # 48 is the max length of table names, this may have collisions but this would only affect to object instantiation that are not really expected (user should give the name of the object instead of relying on the system to generate it)
            # new name as ksp.Dra_n_dom_table_attrname[:40]
            val.make_persistent(name)
        return val

    def __convert_types_to_istorage(self, key, val):
        """ Convert values types to IStorage: 
                np.ndarray --> StorageNumpy,
                set        --> EmbeddedSet 
            TODO: Integrate into make_val_persistent
        """

        if isinstance(val, list):
            vals_istorage = []
            for element in val:
                if isinstance(element, np.ndarray) and not isinstance(element, StorageNumpy):
                    val_istorage = StorageNumpy(element)
                else:
                    val_istorage = element
                vals_istorage.append(val_istorage)

            val = vals_istorage
        elif isinstance(val, np.ndarray) and not isinstance(val, StorageNumpy):
            val = StorageNumpy(val)
        elif isinstance(val, set):
            val = self.__create_embeddedset(key=key, val=val)
        return val

    def close_stream(self):
        '''
        This sends an EOD to the poll of the topic. Typically used after
        sending ALL elements of the dictionary:
            send(k1..)
            send(k2..)
            ...
            send(kN..)
            close_stream()
        '''
        self._hcache.close_stream()

    def send(self, key=None, val=None):
        if key is None and val is None:
            raise NotImplementedError("Send a whole DICTIONARY {}".format(self._name))
        else:
            self.__send_values_kafka(key, val)

    def __send_values_kafka(self, key, val):
        log.debug("__send_values_kafka: key={} value={}".format(key,val))
        if not self._is_persistent:
            raise RuntimeError("'send' operation is only valid on persistent objects")
        if not self._is_stream() :
            raise RuntimeError("current dictionary {} is not a stream".format(self._name))

        if getattr(self,"_topic_name",None) is None:
            self._initialize_stream_capability(self.storage_id)

        if not self._stream_producer_enabled:
            self._hcache.enable_stream_producer()
            self._stream_producer_enabled=True

        tosend=[]
        if not isinstance(val,list):
            val = [val]

        for element in val:
            if isinstance(element, IStorage):
                tosend.append(element.storage_id)
                # Enable stream capability and send it (depends on the object)
                element._initialize_stream_capability(element.storage_id) # TODO this can be removed as it is already done in 'send'
                element.send()
            else:
                tosend.append(element)

        self._hcache.send_event(key, tosend) # Send currrent object list with storageids and basic_types

    def __setitem__(self, key, val):
        """
           Method to insert values in the StorageDict
           Args:
               key: the position of the value that we want to save
               val: the value that we want to save in that position
        """
        oldval = val # DEBUG purposes
        val = self.__convert_types_to_istorage(key, oldval)

        log.debug('SET ITEM %s->%s', key, val)
        if self.storage_id is None:
            dict.__setitem__(self, key, val)
        elif not isinstance(val, EmbeddedSet):
            # Not needed because it is made persistent and inserted to hcache when calling to self.__create_embeddedset
            val = self.__make_val_persistent(val)
            k = self._make_key(key)
            v = self._make_value(val)

            if config.max_cache_size == 0: # If C++ cache is disabled, use python memory
                dict.__setitem__(self,key,val)

            if self._is_stream() :
                self.__send_values_kafka(k,val) # stream values
            self._hcache.put_row(k,v) # ONLY store values in Cassandra

    def poll(self):
        log.debug("StorageDict: POLL ")

        if not self._is_stream():
            raise RuntimeError("Poll on a not streaming object")

        if getattr(self,"_topic_name",None) is None:
            self._initialize_stream_capability(self.storage_id)
        if not self._stream_consumer_enabled:
            self._hcache.enable_stream_consumer()
            self._stream_consumer_enabled=True

        row = self._hcache.poll() # polls any value and caches it

        v=row[-(len(row)-self._k_size):]
        k=row[0:self._k_size]
        log.debug("poll row type {} got row: {}".format(type(row),row))
        log.debug("poll k {} v {}".format(k,v))
        log.debug("poll k type {} ".format(type(k)))

        if len(k)==1 and k[0] is None: # Last item in dictionary sent, exit TODO This only considers single key!
            return self._key_column_builder(*k,*v) # Return None, None (for ALL columns)

        if config.max_cache_size == 0: # If C++ cache is disabled, use python memory
            dict.__setitem__(self, k, v)

        # FIXME : Return  {key, value} instead of {value}
        final_results = []
        for index, col in enumerate(self._columns):
            col_type = col["type"]
            log.debug(" col_type = {}".format(col_type))
            element = v[index]
            if col_type not in basic_types:
                # element is not a built-in type
                info = {"storage_id": element, "tokens": self._build_args.tokens, "class_name": col_type}
                element = build_remotely(info)
                if isinstance(element, StorageNumpy):
                    element._initialize_stream_capability(element.storage_id)
                    element.poll()
                #TODO add code to support polling a StorageObject or a StorageDict

            final_results.append(element)
        return self._key_column_builder(*k,*final_results) # Return Key, Value

    def __len__(self):
        if not self.storage_id:
            return super().__len__()

        self.sync()
        if self._tokens[0][0] == _min_token and self._tokens[-1][1] == _max_token:
            query = f"SELECT COUNT(*) FROM {self._ksp}.{self._table}"
            return self._count_elements(query)

        else:
            keys = []
            for pkey in self._primary_keys:
                template = "'{}'" if pkey["type"] == "text" else "{}"
                keys.append(template.format(pkey["name"]))
            all_keys = ",".join(keys)

            total = 0
            for (token_start, token_end) in self._tokens:
                query = f"SELECT COUNT(*) FROM {self._ksp}.{self._table} " \
                    f"WHERE token({all_keys})>={token_start} AND token({all_keys})<{token_end}"

                total = total + self._count_elements(query)
            return total

    def __repr__(self):
        """
        Overloads the method used by print to show a StorageDict
        Returns: The representation of the data stored in the StorageDict

        """
        to_return = {}
        for item in self.items():
            to_return[item[0]] = item[1]
            if len(to_return) == config.hecuba_print_limit:
                return str(to_return)
        if len(to_return) > 0:
            return str(to_return)
        return ""

    def update(self, other=None, **kwargs):
        """
        Updates the current dict with a new dictionary or set of attr,value pairs
        (those must follow the current dict data model).
        Args:
            other: python dictionary or StorageDict. All key,val values in it will
            be inserted in the current dict.
            **kwargs: set of attr:val pairs, to be treated as key,val and inserted
            in the current dict.
        """
        if other is not None:
            if isinstance(other, StorageDict):
                for k, v in other.items():
                    self[k] = v
            else:
                for k, v in other.items() if isinstance(other, Mapping) else other:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def keys(self):
        """
        Obtains the iterator for the keys of the StorageDict
        Returns:
            if persistent:
                iterkeys(self): list of keys
            if not persistent:
                dict.keys(self)
        """
        if self.storage_id:
            self.sync()
            ik = self._hcache.iterkeys(config.prefetch_size)
            iterator = NamedIterator(ik, self._key_builder, self)
            if self._has_embedded_set:
                iterator = iter(set(iterator))

            return iterator
        else:
            return dict.keys(self)

    def items(self):
        """
        Obtains the iterator for the key,val pairs of the StorageDict
        Returns:
            if persistent:
                NamedItemsIterator(self): list of key,val pairs
            if not persistent:
                dict.items(self)
        """
        if self.storage_id:
            self.sync()
            ik = self._hcache.iteritems(config.prefetch_size)
            iterator = NamedItemsIterator(self._key_builder,
                                          self._column_builder,
                                          self._k_size,
                                          ik,
                                          self)
            if self._has_embedded_set:
                d = defaultdict(set)
                # iteritems has the set values in different rows, this puts all the set values in the same row
                if len(self._get_set_types()) == 1:
                    for row in iterator:
                        d[row.key].add(row.value[0])
                else:
                    for row in iterator:
                        d[row.key].add(tuple(row.value))

                iterator = d.items()

            return iterator
        else:
            return dict.items(self)

    def values(self):
        """
        Obtains the iterator for the values of the StorageDict
        Returns:
            if persistent:
                NamedIterator(self): list of valuesStorageDict
            if not persistent:
                dict.values(self)
        """
        if self.storage_id:
            self.sync()
            if self._has_embedded_set:
                items = self.items()
                return dict(items).values()
            else:
                ik = self._hcache.itervalues(config.prefetch_size)
                return NamedIterator(ik, self._column_builder, self)
        else:
            return dict.values(self)

    def get(self, key, default=None):
        try:
            value = self.__getitem__(key)
        except KeyError:
            value = default
        return value

    def _get_set_types(self):
        """
        Returns a list of tuples (name,type) for the types of the set
        """
        if self._has_embedded_set:
            set_types = [col.get("columns", []) for col in self._columns if isinstance(col, dict)]
            return sum(set_types, [])
        else:
            return []
