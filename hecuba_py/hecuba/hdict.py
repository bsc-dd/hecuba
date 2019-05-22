import uuid
from collections import Iterable, defaultdict
from collections import Mapping
from collections import namedtuple

import numpy as np
from hecuba import config, log, Parser
from hecuba.tools import NamedItemsIterator, NamedIterator
from hecuba.hnumpy import StorageNumpy
from hfetch import Hcache

from IStorage import IStorage, AlreadyPersistentError, _basic_types, _discrete_token_ranges, _extract_ks_tab


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
        if not isinstance(value, Iterable) or isinstance(value, str) or isinstance(value, unicode):
            keys.append(value)
        else:
            keys += list(value)
        return self._father.__setitem__(keys, [])

    def remove(self, value):
        if value in self:
            keys = self._keys[:]
            if not isinstance(value, Iterable) or isinstance(value, str) or isinstance(value, unicode):
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
                if not isinstance(value, Iterable) or isinstance(value, str) or isinstance(value, unicode):
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
        if not isinstance(value, Iterable) or isinstance(value, str) or isinstance(value, unicode):
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


class StorageDict(dict, IStorage):
    # """
    # Object used to access data from workers.
    # """

    args_names = ["name", "primary_keys", "columns", "tokens", "storage_id", "indexed_on", "class_name", "built_remotely"]
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

    def __init__(self, name=None, primary_keys=None, columns=None, tokens=None,
                 storage_id=None, indexed_on=None, built_remotely=False, **kwargs):
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

        super(StorageDict, self).__init__(**kwargs)
        self._is_persistent = False
        self._built_remotely = built_remotely
        log.debug("CREATED StorageDict(%s,%s,%s,%s,%s,%s)", primary_keys, columns, name, tokens, storage_id, kwargs)

        if tokens is None:
            log.info('using all tokens')
            tokens = map(lambda a: a.value, config.cluster.metadata.token_map.ring)
            self._tokens = _discrete_token_ranges(tokens)
        else:
            self._tokens = tokens

        self._storage_id = storage_id

        if self.__doc__ is not None:
            self._persistent_props = self._parse_comments(self.__doc__)
            self._primary_keys = self._persistent_props['primary_keys']
            self._columns = self._persistent_props['columns']
            self._indexed_on = self._persistent_props.get('indexed_on', indexed_on)
        else:
            self._primary_keys = primary_keys
            set_pks = []
            normal_columns = []
            for column_name, column_type in columns:
                if column_name.find("_set_") != -1:
                    set_pks.append((column_name.replace("_set_", ""), column_type))
                else:
                    normal_columns.append((column_name, column_type))
            if set_pks:
                self._columns = [{"type": "set", "columns": set_pks}]
            else:
                self._columns = columns
            self._indexed_on = indexed_on

        self._has_embedded_set = False
        build_column = []
        columns = []
        for col in self._columns:
            if isinstance(col, dict):
                types = col["columns"]
                if col["type"] == "set":
                    self._has_embedded_set = True
                    for t in types:
                        build_column.append(("_set_" + t[0], t[1]))
                else:
                    build_column.append((col["name"], col["type"]))
                columns.append(col)
            else:
                columns.append({"type": col[1], "name": col[0]})
                build_column.append(col)

        self._columns = columns[:]
        self._primary_keys = [{"type": key[1], "name": key[0]} if isinstance(key, tuple) else key
                              for key in self._primary_keys]
        build_keys = [(key["name"], key["type"]) for key in self._primary_keys]

        key_names = [col["name"] for col in self._primary_keys]
        column_names = [col["name"] for col in self._columns]

        self._item_builder = namedtuple('row', key_names + column_names)

        if len(key_names) > 1:
            self._key_builder = namedtuple('row', key_names)
        else:
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

        if build_column is None:
            build_column = self._columns[:]

        self._build_args = self.args(None, build_keys, build_column, self._tokens,
                                     self._storage_id, self._indexed_on, class_name, built_remotely)

        if name:
            self.make_persistent(name)

    def __eq__(self, other):
        """
        Method to compare a StorageDict with another one.
        Args:
            other: StorageDict to be compared with.
        Returns:
            boolean (true - equals, false - not equals).
        """
        return self._storage_id == other._storage_id and self._tokens == other.token_ranges and \
               self._table == other.table_name and self._ksp == other.keyspace

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
        if not self._is_persistent:
            return dict.__contains__(self, key)
        else:
            try:
                # TODO we should save this value in a cache
                self._hcache.get_row(self._make_key(key))
                return True
            except Exception as ex:
                log.warn("persistentDict.__contains__ ex %s", ex)
                return False

    def _make_key(self, key):
        """
        Method used to pass the key data to the StorageDict cache in a proper way.
        Args:
            key: the data that needs to get the correct format
        """
        if isinstance(key, str) or isinstance(key, unicode) or not isinstance(key, Iterable):
            if len(self._primary_keys) == 1:
                if isinstance(key, unicode):
                    return [key.encode('ascii', 'ignore')]
                return [key]
            else:
                raise Exception('missing a primary key')

        if isinstance(key, Iterable) and len(key) == len(self._primary_keys):
            return list(key)
        elif self._has_embedded_set and isinstance(key, Iterable) and len(key) == (
                len(self._primary_keys) + len(self._get_set_types())):
            return list(key)
        else:
            raise Exception('wrong primary key')

    @staticmethod
    def _make_value(value):
        """
        Method used to pass the value data to the StorageDict cache in a proper way.
        Args:
            value: the data that needs to get the correct format
        """
        if issubclass(value.__class__, IStorage):
            return [uuid.UUID(value.getID())]
        elif isinstance(value, str) or not isinstance(value, Iterable) or isinstance(value, np.ndarray):
            return [value]
        elif isinstance(value, unicode):
            return [value.encode('ascii', 'ignore')]
        elif isinstance(value, tuple):
            return [value]
        elif isinstance(value, Iterable):
            val = []
            for v in value:
                if isinstance(v, IStorage):
                    val.append(uuid.UUID(v.getID()))
                else:
                    val.append(v)
            return val
        else:
            return list(value)

    def keys(self):
        """
        This method return a list of all the keys of the StorageDict.
        Returns:
          list: a list of keys
        """
        return [i for i in self.iterkeys()]

    def values(self):
        """
        This method return a list of all the values of the StorageDict.
        Returns:
          list: a list of values
        """
        return [i for i in self.itervalues()]

    def items(self):
        """
        This method return a list of all the key-value pairs of the StorageDict.
        Returns:
          list: a list of key-value pairs
        """
        return [i for i in self.iteritems()]

    def __iter__(self):
        """
        Method that overloads the python dict basic iteration, which returns
        an iterator over the dictionary keys.
        """
        return self.iterkeys()

    def make_persistent(self, name):
        """
        Method to transform a StorageDict into a persistent object.
        This will make it use a persistent DB as the main location
        of its data.
        Args:
            name:
        """
        if self._is_persistent:
            raise AlreadyPersistentError("This StorageDict is already persistent [Before:{}.{}][After:{}]",
                                         self._ksp, self._table, name)

        # Update local StorageDict metadata
        self._is_persistent = True
        (self._ksp, self._table) = _extract_ks_tab(name)

        if self._storage_id is None:
            self._storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, self._ksp + '.' + self._table)
        self._build_args = self._build_args._replace(storage_id=self._storage_id, name=self._ksp + "." + self._table)

        # Prepare data
        persistent_keys = [(key["name"], "tuple<" + ",".join(key["columns"]) + ">") if key["type"] == "tuple"
                           else (key["name"], key["type"]) for key in self._primary_keys] + self._get_set_types()
        persistent_values = []
        if not self._has_embedded_set:
            for col in self._columns:
                if col["type"] == "tuple":
                    persistent_values.append({"name": col["name"], "type": "tuple<" + ",".join(col["columns"]) + ">"})
                elif col["type"] not in _basic_types:
                    persistent_values.append({"name": col["name"], "type": "uuid"})
                else:
                    persistent_values.append({"name": col["name"], "type": col["type"]})

        key_names = [col[0] if isinstance(col, tuple) else col["name"] for col in persistent_keys]

        if config.id_create_schema == -1 and not self._built_remotely:
            query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (self._ksp, config.replication)
            try:
                log.debug('MAKE PERSISTENCE: %s', query_keyspace)
                config.session.execute(query_keyspace)
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
                config.session.execute(query_table)
            except Exception as ex:
                log.warn("Error creating the StorageDict table: %s %s", query_table, ex)
                raise ex

            if hasattr(self, '_indexed_on') and self._indexed_on is not None:
                index_query = 'CREATE CUSTOM INDEX IF NOT EXISTS ' + self._table + '_idx ON '
                index_query += self._ksp + '.' + self._table + ' (' + str.join(',', self._indexed_on) + ') '
                index_query += "using 'es.bsc.qbeast.index.QbeastIndex';"
                try:
                    config.session.execute(index_query)
                except Exception as ex:
                    log.error("Error creating the Qbeast custom index: %s %s", index_query, ex)
                    raise ex
                trigger_query = "CREATE TRIGGER IF NOT EXISTS %s%s_qtr ON %s.%s USING 'es.bsc.qbeast.index.QbeastTrigger';" % \
                                (self._ksp, self._table, self._ksp, self._table)
                try:
                    config.session.execute(trigger_query)
                except Exception as ex:
                    log.error("Error creating the Qbeast trigger: %s %s", trigger_query, ex)
                    raise ex

        self._hcache_params = (self._ksp, self._table,
                               self._storage_id,
                               self._tokens, key_names, persistent_values,
                               {'cache_size': config.max_cache_size,
                                'writer_par': config.write_callbacks_number,
                                'writer_buffer': config.write_buffer_size,
                                'timestamped_writes' : config.timestamped_writes})
        log.debug("HCACHE params %s", self._hcache_params)
        self._hcache = Hcache(*self._hcache_params)

        # Storing all in-memory values to cassandra
        for key, value in dict.iteritems(self):
            if issubclass(value.__class__, IStorage):
                # new name as ksp.table_valuename, where valuename is either defined by the user or set by hecuba
                if not value._is_persistent:
                    val_name = self._ksp + '.' + self._table + '_' + self._columns[0]["name"]
                    value.make_persistent(val_name)
                value = value._storage_id
            self._hcache.put_row(self._make_key(key), self._make_value(value))

        super(StorageDict, self).clear()

        self._store_meta(self._build_args)

    def stop_persistent(self):
        """
        Method to turn a StorageDict into non-persistent.
        """
        log.debug('STOP PERSISTENCE: %s', self._table)
        self._is_persistent = False
        self._hcache = None

    def delete_persistent(self):
        """
        Method to empty all data assigned to a StorageDict.
        """
        query = "TRUNCATE TABLE %s.%s;" % (self._ksp, self._table)
        log.debug('DELETE PERSISTENT: %s', query)
        config.session.execute(query)

    def __delitem__(self, key):
        """
        Method to delete a specific entry in the dict in the key position.
        Args:
            key: position of the entry that we want to delete
        """
        if not self._is_persistent:
            dict.__delitem__(self, key)
        elif self._has_embedded_set:
            self._hcache.delete_row(key)
        else:
            self._hcache.delete_row([key])

    def __create_embeddedset(self, key, val=None):
        if not isinstance(key, Iterable) or isinstance(key, str) or isinstance(key, unicode):
            return EmbeddedSet(self, [key], val)
        else:
            return EmbeddedSet(self, list(key), val)

    def __getitem__(self, key):
        """
        If the object is persistent, each request goes to the hfetch.
        Args:
             key: the dictionary key
        Returns
             item: value found in position key
        """
        log.debug('GET ITEM %s', key)

        if not self._is_persistent:
            return dict.__getitem__(self, key)
        elif self._has_embedded_set:
            return self.__create_embeddedset(key=key)
        else:
            # Returns always a list with a single entry for the key
            persistent_result = self._hcache.get_row(self._make_key(key))
            log.debug("GET ITEM %s[%s]", persistent_result, persistent_result.__class__)

            # we need to transform UUIDs belonging to IStorage objects and rebuild them
            final_results = []
            for index, col in enumerate(self._columns):
                name = col["name"]
                col_type = col["type"]
                element = persistent_result[index]
                if col_type not in _basic_types:
                    # element is not a built-in type
                    table_name = self._ksp + '.' + self._table + '_' + name
                    info = {"name": table_name, "tokens": self._build_args.tokens, "storage_id": uuid.UUID(element),
                            "class_name": col_type}
                    element = IStorage.build_remotely(info)

                final_results.append(element)

            if self._column_builder is not None:
                return self._column_builder(*final_results)
            else:
                return final_results[0]

    def __make_val_persistent(self, val, col=0):
        if isinstance(val, StorageDict):
            for k, element in val.iteritems():
                val[k] = self.__make_val_persistent(element)
        elif isinstance(val, list):
            for index, element in enumerate(val):
                val[index] = self.__make_val_persistent(element, index)
        if isinstance(val, IStorage) and not val._is_persistent:
            val._storage_id = uuid.uuid4()
            attribute = self._columns[col]["name"]
            count = self._count_name_collision(attribute)
            if count == 0:
                name = self._ksp + "." + self._table + "_" + attribute
            else:
                name = self._ksp + "." + self._table + "_" + attribute + "_" + str(count - 1)
            # new name as ksp+table+obj_class_name
            val.make_persistent(name)
        return val

    def __setitem__(self, key, val):
        """
           Method to insert values in the StorageDict
           Args:
               key: the position of the value that we want to save
               val: the value that we want to save in that position
        """
        if isinstance(val, list):
            vals_istorage = []
            for element in val:
                if isinstance(element, np.ndarray):
                    val_istorage = StorageNumpy(element)
                else:
                    val_istorage = element
                vals_istorage.append(val_istorage)

            val = vals_istorage
        elif isinstance(val, np.ndarray):
            val = StorageNumpy(val)
        elif isinstance(val, set):
            val = self.__create_embeddedset(key=key, val=val)

        log.debug('SET ITEM %s->%s', key, val)
        if not self._is_persistent:
            dict.__setitem__(self, key, val)
        elif not isinstance(val, EmbeddedSet):
            # Not needed because it is made persistent and inserted to hcache when calling to self.__create_embeddedset
            val = self.__make_val_persistent(val)
            self._hcache.put_row(self._make_key(key), self._make_value(val))

    def __repr__(self):
        """
        Overloads the method used by print to show a StorageDict
        Returns: The representation of the data stored in the StorageDict

        """
        to_return = {}
        for item in self.iteritems():
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
                for k, v in other.iteritems():
                    self[k] = v
            else:
                for k, v in other.items() if isinstance(other, Mapping) else other:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def iterkeys(self):
        """
        Obtains the iterator for the keys of the StorageDict
        Returns:
            if persistent:
                iterkeys(self): list of keys
            if not persistent:
                dict.iterkeys(self)
        """
        if self._is_persistent:
            ik = self._hcache.iterkeys(config.prefetch_size)
            iterator = NamedIterator(ik, self._key_builder, self)
            if self._has_embedded_set:
                iterator = iter(set(iterator))

            return iterator
        else:
            return dict.iterkeys(self)

    def iteritems(self):
        """
        Obtains the iterator for the key,val pairs of the StorageDict
        Returns:
            if persistent:
                NamedItemsIterator(self): list of key,val pairs
            if not persistent:
                dict.iteritems(self)
        """
        if self._is_persistent:
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
                    map(lambda row: d[row[0]].add(row[1][0]), iterator)
                else:
                    map(lambda row: d[row[0]].add(tuple(row[1])), iterator)
                iterator = d.iteritems()

            return iterator
        else:
            return dict.iteritems(self)

    def itervalues(self):
        """
        Obtains the iterator for the values of the StorageDict
        Returns:
            if persistent:
                NamedIterator(self): list of valuesStorageDict
            if not persistent:
                dict.itervalues(self)
        """
        if self._is_persistent:
            if self._has_embedded_set:
                iteritems = self.iteritems()
                return dict(iteritems).itervalues()
            else:
                ik = self._hcache.itervalues(config.prefetch_size)
                return NamedIterator(ik, self._column_builder, self)
        else:
            return dict.itervalues(self)

    def get(self, key, default):
        try:
            value = self.__getitem__(key)
        except KeyError:
            value = default
        return value

    def _get_set_types(self):
        if self._has_embedded_set:
            set_types = [col.get("columns", []) for col in self._columns if isinstance(col, dict)]
            return sum(set_types, [])
        else:
            return []
