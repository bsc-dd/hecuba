from collections import Iterable
from collections import namedtuple
from collections import Mapping
from types import NoneType
from hfetch import Hcache
from IStorage import IStorage, AlreadyPersistentError
from hecuba import config, log
from hecuba.hnumpy import StorageNumpy
import uuid
import re
import numpy as np


class NamedIterator:
    # Class that allows to iterate over the keys or the values of a dict
    def __init__(self, hiterator, builder, father):
        self.hiterator = hiterator
        self.builder = builder
        self._storage_father = father

    def __iter__(self):
        return self

    def next(self):
        n = self.hiterator.get_next()
        if self.builder is not None:
            return self.builder(*n)
        else:
            return n[0]


class NamedItemsIterator:
    # Class that allows to iterate over the keys and the values of a dict
    builder = namedtuple('row', 'key, value')

    def __init__(self, key_builder, column_builder, k_size, hiterator, father):
        self.key_builder = key_builder
        self.k_size = k_size
        self.column_builder = column_builder
        self.hiterator = hiterator
        self._storage_father = father

    def __iter__(self):
        return self

    def next(self):
        n = self.hiterator.get_next()
        if self.key_builder is None:
            k = n[0]
        else:
            k = self.key_builder(*n[0:self.k_size])
        if self.column_builder is None:
            v = n[self.k_size]
            if isinstance(v, unicode):
                v = str(v)
        else:
            v = self.column_builder(*n[self.k_size:])
        return self.builder(k, v)


class StorageDict(dict, IStorage):
    # """
    # Object used to access data from workers.
    # """

    args_names = ["name", "primary_keys", "columns", "tokens", "storage_id", "indexed_on", "class_name"]
    args = namedtuple('StorageDictArgs', args_names)
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba.istorage'
                                                  '(storage_id, class_name, name, tokens, '
                                                  'primary_keys, columns, indexed_on)'
                                                  'VALUES (?,?,?,?,?,?,?)')

    @staticmethod
    def build_remotely(result):
        """
        Launches the StorageDict.__init__ from the api.getByID
        Args:
            result: a namedtuple with all  the information needed to create again the StorageDict
        """
        log.debug("Building Storage dict with %s", result)

        return StorageDict(result.name,
                           result.primary_keys,
                           result.columns,
                           result.tokens,
                           result.storage_id,
                           result.indexed_on
                           )

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

    def __init__(self, name="", primary_keys=None, columns=None, tokens=None,
                 storage_id=None, indexed_args=None, **kwargs):
        """
        Creates a new StorageDict.

        Args:
            name (string): the name of the collection/table (keyspace is optional)
            primary_keys (list(tuple)): a list of (key,type) primary keys (primary + clustering).
            columns (list(tuple)): a list of (key,type) columns
            tokens (list): list of tokens
            storage_id (string): the storage id identifier
            indexed_args (list): values that will be used as index
            kwargs: other parameters
        """

        super(StorageDict, self).__init__(**kwargs)
        self._is_persistent = False
        log.debug("CREATED StorageDict(%s,%s,%s,%s,%s,%s)", primary_keys, columns, name, tokens, storage_id, kwargs)

        if tokens is None:
            log.info('using all tokens')
            tokens = map(lambda a: a.value, config.cluster.metadata.token_map.ring)
            self._tokens = IStorage._discrete_token_ranges(tokens)
        else:
            self._tokens = tokens

        self._storage_id = storage_id

        if self.__doc__ is not None:
            self._persistent_props = self._parse_comments(self.__doc__)
            self._primary_keys = self._persistent_props[self.__class__.__name__]['primary_keys']
            self._columns = self._persistent_props[self.__class__.__name__]['columns']
            try:
                self._indexed_args = self._persistent_props[self.__class__.__name__]['indexed_values']
            except KeyError:
                self._indexed_args = indexed_args
        else:
            self._primary_keys = primary_keys
            self._columns = columns
            self._indexed_args = indexed_args

        key_names = [pkname for (pkname, dt) in self._primary_keys]
        column_names = [colname for (colname, dt) in self._columns]
        self._item_builder = namedtuple('row', key_names + column_names)

        if len(key_names) > 1:
            self._key_builder = namedtuple('row', key_names)
        else:
            self._key_builder = None
        if len(column_names) > 1:
            self._column_builder = namedtuple('row', column_names)
        else:
            self._column_builder = None

        self._k_size = len(key_names)

        class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)
        self._build_args = self.args(name, self._primary_keys, self._columns, self._tokens,
                                     self._storage_id, self._indexed_args, class_name)

        if name:
            self.make_persistent(name)
        else:
            self._is_persistent = False

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

    _dict_case = re.compile('.*@TypeSpec + *< *< *([\w:, ]+)+ *> *, *([\w+:., <>]+) *>')
    _tuple_case = re.compile('.*@TypeSpec +(\w+) +tuple+ *< *([\w, +]+) *>')
    _index_vars = re.compile('.*@Index_on *([A-z0-9, ]+)')
    _other_case = re.compile(' *(\w+) *< *([\w, +]+) *>')

    @classmethod
    def _parse_comments(self, comments):
        """
            Parses de comments in a class file to save them in the class information
            Args:
                comments: the comment in the class file
            Returns:
                this: a structure with all the information of the comment
        """
        this = {}
        for line in comments.split('\n'):
            m = StorageDict._dict_case.match(line)
            if m is not None:
                # Matching @TypeSpec of a dict
                dict_keys, dict_values = m.groups()
                primary_keys = []
                for ind, key in enumerate(dict_keys.split(",")):
                    key = key.replace(' ', '')
                    match = IStorage._data_type.match(key)
                    if match is not None:
                        # an IStorage with a name
                        name, value = match.groups()
                    elif ':' in key:
                        raise SyntaxError
                    else:
                        name = "key" + str(ind)
                        value = key

                    name = name.replace(' ', '')
                    value = value.replace(' ', '')
                    primary_keys.append((name, StorageDict._conversions[value]))
                dict_values = dict_values.replace(' ', '')
                if dict_values.startswith('dict'):
                    n = IStorage._sub_dict_case.match(dict_values[4:])
                    # Matching @TypeSpec of a sub dict
                    dict_keys2, dict_values2 = n.groups()
                    primary_keys2 = []
                    for ind, key in enumerate(dict_keys2.split(",")):
                        try:
                            name, value = IStorage._data_type.match(key).groups()
                        except ValueError:
                            if ':' in key:
                                raise SyntaxError
                            else:
                                name = "key" + str(ind)
                                value = key
                        name = name.replace(' ', '')
                        primary_keys2.append((name, StorageDict._conversions[value]))
                    columns2 = []
                    dict_values2 = dict_values2.replace(' ', '')
                    if dict_values2.startswith('tuple'):
                        dict_values2 = dict_values2[6:]
                    for ind, val in enumerate(dict_values2.split(",")):
                        try:
                            name, value = IStorage._data_type.match(val).groups()
                        except ValueError:
                            if ':' in key:
                                raise SyntaxError
                            else:
                                name = "val" + str(ind)
                                value = val
                        columns2.append((name, StorageDict._conversions[value]))
                    columns = {
                        'type': 'dict',
                        'primary_keys': primary_keys2,
                        'columns': columns2}
                elif dict_values.startswith('tuple'):
                    n = IStorage._sub_tuple_case.match(dict_values[5:])
                    tuple_values = list(n.groups())[0]
                    columns = []
                    for ind, val in enumerate(tuple_values.split(",")):
                        try:
                            name, value = val.split(':')
                        except ValueError:
                            if ':' in key:
                                raise SyntaxError
                            else:
                                name = "val" + str(ind)
                                value = val
                        name = name.replace(' ', '')
                        columns.append((name, StorageDict._conversions[value]))
                else:
                    columns = []
                    for ind, val in enumerate(dict_values.split(",")):
                        match = IStorage._data_type.match(val)
                        if match is not None:
                            # an IStorage with a name
                            name, value = match.groups()
                        elif ':' in val:
                            name, value = IStorage._so_data_type.match(val).groups()
                        else:
                            name = "val" + str(ind)
                            value = val
                        name = name.replace(' ', '')
                        try:
                            columns.append((name, StorageDict._conversions[value]))
                        except KeyError:
                            columns.append((name, value))
                name = str(self).replace('\'>', '').split('.')[-1]
                if self.__class__.__name__ in this:
                    this[name].update({'type': 'dict', 'primary_keys': primary_keys, 'columns': columns})
                else:
                    this[name] = {
                        'type': 'dict',
                        'primary_keys': primary_keys,
                        'columns': columns}
            m = StorageDict._index_vars.match(line)
            if m is not None:
                name = str(self).replace('\'>', '').split('.')[-1]
                indexed_values = m.groups()
                indexed_values = indexed_values.replace(' ', '').split(',')
                if name in this:
                    this[name].update({'indexed_values': indexed_values})
                else:
                    this[name] = {'indexed_values': indexed_values}
        return this

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
                raise ex

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
        self._is_persistent = True
        (self._ksp, self._table) = self._extract_ks_tab(name)

        if self._storage_id is None:
            self._storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, self._ksp + '.' + self._table)
        self._build_args = self._build_args._replace(storage_id=self._storage_id, name=self._ksp + "." + self._table)
        self._store_meta(self._build_args)
        if config.id_create_schema == -1:
            query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (self._ksp, config.replication)
            try:
                log.debug('MAKE PERSISTENCE: %s', query_keyspace)
                config.session.execute(query_keyspace)
            except Exception as ex:
                log.warn("Error creating the StorageDict keyspace %s, %s", (query_keyspace), ex)
                raise ex

        all_columns = self._primary_keys + self._columns
        for ind, entry in enumerate(all_columns):
            n = StorageDict._other_case.match(entry[1])
            if n is not None:
                iter_type, intra_type = n.groups()
            else:
                iter_type = entry[1]
            if iter_type not in IStorage._basic_types:
                all_columns[ind] = entry[0], 'uuid'

        pks = map(lambda a: a[0], self._primary_keys)
        query_table = "CREATE TABLE IF NOT EXISTS %s.%s (%s, PRIMARY KEY (%s));" \
                      % (self._ksp,
                         self._table,
                         ",".join("%s %s" % tup for tup in all_columns),
                         str.join(',', pks))
        try:
            log.debug('MAKE PERSISTENCE: %s', query_table)
            config.session.execute(query_table)
        except Exception as ex:
            log.warn("Error creating the StorageDict table: %s %s", query_table, ex)
            raise ex
        key_names = map(lambda a: a[0].encode('UTF8'), self._primary_keys)
        values_names = self._columns

        self._hcache_params = (self._ksp, self._table,
                               self._storage_id,
                               self._tokens, key_names, map(lambda x: {"name": x[0], "type": x[1]}, values_names),
                               {'cache_size': config.max_cache_size,
                                'writer_par': config.write_callbacks_number,
                                'write_buffer': config.write_buffer_size})
        log.debug("HCACHE params %s", self._hcache_params)
        self._hcache = Hcache(*self._hcache_params)
        # Storing all in-memory values to cassandra
        for key, value in dict.iteritems(self):
            if issubclass(value.__class__, IStorage):
                # new name as ksp.table_valuename, where valuename is either defined by the user or set by hecuba
                val_name = self._ksp + '.' + self._table + '_' + self._columns[0][0]
                value.make_persistent(val_name)
                value = value._storage_id
            self._hcache.put_row(self._make_key(key), self._make_value(value))
        if hasattr(self, '_indexed_args') and self._indexed_args is not None:
            index_query = 'CREATE CUSTOM INDEX IF NOT EXISTS ' + self._table + '_idx ON '
            index_query += self._ksp + '.' + self._table + ' (' + str.join(',', self._indexed_args) + ') '
            index_query += "using 'es.bsc.qbeast.index.QbeastIndex';"
            try:
                config.session.execute(index_query)
            except Exception as ex:
                log.error("Error creating the Qbeast custom index: %s %s", index_query, ex)
                raise ex

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
        else:
            self._hcache.delete_row([key])

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
        else:
            # Returns always a list with a single entry for the key
            persistent_result = self._hcache.get_row(self._make_key(key))
            log.debug("GET ITEM %s[%s]", persistent_result, persistent_result.__class__)

            # we need to transform UUIDs belonging to IStorage objects and rebuild them
            final_results = []
            for index, (name, col_type) in enumerate(self._columns):
                element = persistent_result[index]
                if col_type not in IStorage._basic_types:
                    # element is not a built-in type
                    table_name = self._ksp + '.' + self._table + '_' + name
                    element = self._build_istorage_obj(name=table_name, type=col_type, tokens=self._build_args.tokens,
                                                       storage_id=uuid.UUID(element))
                final_results.append(element)

            if self._column_builder is not None:
                return self._column_builder(*final_results)
            else:
                return final_results[0]

    def __setitem__(self, key, val):
        """
           Method to insert values in the StorageDict
           Args:
               key: the position of the value that we want to save
               val: the value that we want to save in that position
        """
        if isinstance(val, np.ndarray):
            val = StorageNumpy(val)
        log.debug('SET ITEM %s->%s', key, val)
        if not config.hecuba_type_checking:
            if not self._is_persistent:
                dict.__setitem__(self, key, val)
            else:
                if isinstance(val, IStorage) and not val._is_persistent:
                    attribute = self._columns[0][0]
                    count = self._count_name_collision(attribute)
                    # new name as ksp+table+obj_class_name
                    val.make_persistent(self._ksp + '.' + self._table + "_" + attribute + "_" + str(count))
                self._hcache.put_row(self._make_key(key), self._make_value(val))
        else:
            if isinstance(val, Iterable) and not isinstance(val, str):
                col_types = map(lambda x: IStorage._conversions[x.__class__.__name__], val)
                spec_col_types = map(lambda x: x[1], self._columns)
                for idx, value in enumerate(spec_col_types):
                    if value == 'double':
                        spec_col_types[idx] = 'float'
            else:
                col_types = IStorage._conversions[val.__class__.__name__]
                spec_col_types = map(lambda x: x[1], self._columns)[0]
                if spec_col_types == 'double':
                    spec_col_types = 'float'
            if isinstance(key, Iterable) and not isinstance(key, str):
                key_types = map(lambda x: IStorage._conversions[x.__class__.__name__], key)
                spec_key_types = map(lambda x: x[1], self._primary_keys)
                for idx, value in enumerate(spec_key_types):
                    if value == 'double':
                        spec_key_types[idx] = 'float'
            else:
                key_types = IStorage._conversions[key.__class__.__name__]
                spec_key_types = map(lambda x: x[1], self._primary_keys)[0]
                if spec_key_types == 'double':
                    spec_key_types = 'float'
            if (col_types == spec_col_types):
                if (key_types == spec_key_types):
                    if not self._is_persistent:
                        dict.__setitem__(self, key, val)
                    else:
                        self._hcache.put_row(self._make_key(key), self._make_value(val))
                else:
                    raise KeyError
            else:
                raise ValueError

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
            return NamedIterator(ik, self._key_builder, self)
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
            return NamedItemsIterator(self._key_builder,
                                      self._column_builder,
                                      self._k_size,
                                      ik,
                                      self)
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
            ik = self._hcache.itervalues(config.prefetch_size)
            return NamedIterator(ik, self._column_builder, self)
        else:
            return dict.itervalues(self)

    def keys(self):
        return [i for i in self.iterkeys()]

    def values(self):
        return [i for i in self.itervalues()]

    def items(self):
        return [i for i in self.iteritems()]

    def get(self, key, default):
        try:
            value = self.__getitem__(key)
        except KeyError:
            value = default
        return value
