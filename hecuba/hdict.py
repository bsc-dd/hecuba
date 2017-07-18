# author: G. Alomar
from collections import Iterable
from collections import namedtuple
from types import NoneType
from hfetch import Hcache
from IStorage import IStorage
from hecuba import config, log
import uuid
import re


class NamedIterator:
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
        Launches the Block.__init__ from the api.getByID
        Args:
            result: a namedtuple with all  the information needed to create again the block
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
                 storage_id=None, indexed_args=[], **kwargs):
        """
        Creates a new block.

        Args:
            table_name (string): the name of the collection/table
            keyspace_name (string): name of the Cassandra keyspace.
            primary_keys (list(tuple)): a list of (key,type) primary keys (primary + clustering).
            columns (list(tuple)): a list of (key,type) columns
            tokens (list): list of tokens
            storage_id (string): the storage id identifier
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

        class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)
        self._build_args = self.args(name, primary_keys, columns, self._tokens,
                                     self._storage_id, indexed_args, class_name)

        if self.__doc__ is not None:
            self._persistent_props = self._parse_comments(self.__doc__)
            self._primary_keys = self._persistent_props[self.__class__.__name__]['primary_keys']
            self._columns = self._persistent_props[self.__class__.__name__]['columns']
            try:
                self._indexed_args = self._persistent_props[self.__class__.__name__]['indexed_values']
            except:
                pass
        else:
            self._primary_keys = primary_keys
            self._columns = columns
            self._indexed_args = indexed_args

        key_names = map(lambda a: a[0], self._primary_keys)
        column_names = map(lambda a: a[0], self._columns)
        self._item_builder = namedtuple('row', map(lambda a: a[0], self._primary_keys + self._columns))

        if len(key_names) > 1:
            self._key_builder = namedtuple('row', key_names)
        else:
            self._key_builder = None
        if len(column_names) > 1:
            self._column_builder = namedtuple('row', column_names)
        else:
            self._column_builder = None

        self._k_size = len(key_names)

        if name is not None:
            self.make_persistent(name)
        else:
            self._is_persistent = False

    def __eq__(self, other):
        return self._storage_id == other._storage_id and \
               self._tokens == other.token_ranges \
               and self._table == other.table_name and self._ksp == other.keyspace

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
        if not self._is_persistent:
            return dict.__contains__(self, key)
        else:
            try:
                self._hcache.get_row(self._make_key(key))
                return True
            except Exception as e:
                print "Exception in persistentDict.__contains__:", e
                return False

    def _make_key(self, key):
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
        if isinstance(value, str) or not isinstance(value, Iterable):
            return [value]
        elif isinstance(value, unicode):
            return [value.encode('ascii', 'ignore')]
        else:
            return list(value)

    def keys(self):
        """
        This method return a list of all the keys of the PersistentDict.
        Returns:
          list: a list of keys
        """
        return [i for i in self.iterkeys()]

    def values(self):
        """
        This method return a list of all the values of the PersistentDict.
        Returns:
          list: a list of values
        """
        return [i for i in self.itervalues()]

    def __iter__(self):
        return self.iterkeys()

    def make_persistent(self, name):
        self._is_persistent = True
        (self._ksp, self._table) = self._extract_ks_tab(name)

        if self._storage_id is None:
            self._storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, self._ksp + '.' + self._table)
        self._build_args = self._build_args._replace(storage_id=self._storage_id, name=self._ksp + "." + self._table)
        self._store_meta(self._build_args)
        if config.id_create_schema == -1:
            query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = {'class': 'SimpleStrategy'," \
                             "'replication_factor': %d }" % (self._ksp, config.repl_factor)
            if query_keyspace not in config.create_cache:
                try:
                    config.create_cache.add(query_keyspace)
                    log.debug('MAKE PERSISTENCE: %s', query_keyspace)
                    config.session.execute(query_keyspace)
                except Exception as ex:
                    print "Error creating the StorageDict keyspace:", query_keyspace, ex

        columns = map(lambda a: a, self._primary_keys + self._columns)
        for ind, entry in enumerate(columns):
            n = StorageDict._other_case.match(entry[1])
            if n is not None:
                iter_type, intra_type = n.groups()
            else:
                iter_type = entry[1]
            if iter_type not in IStorage._valid_types:
                class_name, module = IStorage.process_path(entry[1])
                mod = __import__(module, globals(), locals(), [class_name], 0)
                so = getattr(mod, class_name)(entry[0])
                setattr(self, entry[0], so)
                columns[ind] = entry[0], 'uuid'

        pks = map(lambda a: a[0], self._primary_keys)
        query_table = "CREATE TABLE IF NOT EXISTS %s.%s (%s, PRIMARY KEY (%s));" \
                      % (self._ksp,
                         self._table,
                         ",".join("%s %s" % tup for tup in columns),
                         str.join(',', pks))
        try:
            log.debug('MAKE PERSISTENCE: %s', query_table)
            config.session.execute(query_table)
        except Exception as ex:
            log.warn("Error creating the StorageDict table: %s %s", query_table, ex)
            raise ex
        key_names = map(lambda a: a[0].encode('UTF8'), self._primary_keys)
        column_names = map(lambda a: a[0].encode('UTF8'), self._columns)
        tknp = "token(%s)" % key_names[0]
        self._hcache_params = (self._ksp, self._table,
                               "WHERE %s>=? AND %s<?;" % (tknp, tknp),
                               self._tokens, key_names, column_names,
                               {'cache_size': config.max_cache_size,
                                'writer_par': config.write_callbacks_number,
                                'write_buffer': config.write_buffer_size})
        log.debug("HCACHE params %s", self._hcache_params)
        self._hcache = Hcache(*self._hcache_params)
        # Storing all in-memory values to cassandra
        for key, value in dict.iteritems(self):
            self._hcache.put_row(self._make_key(key), self._make_value(value))
        if hasattr(self, '_indexed_args') and self._indexed_args is not None:
            index_query = 'CREATE CUSTOM INDEX IF NOT EXISTS ' + str(self._table) + '_idx ON '
            index_query += str(self._ksp) + '.' + str(self._table) + ' (' + str.join(',', self._indexed_args) + ') '
            index_query += 'using \'es.bsc.qbeast.index.QbeastIndex\';'
            try:
                config.session.execute(index_query)
            except Exception as ex:
                log.error("Error creating the Qbeast custom index: %s %s", index_query, ex)

    def stop_persistent(self):
        log.debug('STOP PERSISTENCE: %s', self._table)
        self._is_persistent = False
        self._hcache = None

    def delete_persistent(self):
        query = "TRUNCATE TABLE %s.%s;" % (self._ksp, self._table)
        log.debug('DELETE PERSISTENCE: %s', query)
        config.session.execute(query)

    def __getitem__(self, key):
        """
        If the object is persistent, each request goes to the prefetcher.
        Args:
             key: the dictionary key
        Returns
             item: value found in position key
        """
        log.debug('GET ITEM %s', key)

        if not self._is_persistent:
            to_return = dict.__getitem__(self, key)
            return to_return
        else:
            cres = self._hcache.get_row(self._make_key(key))
            log.debug("GET ITEM %s[%s]", cres, cres.__class__)

            if issubclass(cres.__class__, NoneType):
                return None
            elif self._column_builder is not None:
                if len(cres) > 0 and isinstance(cres[0], list):
                    return [self._column_builder(*row) for row in cres]
                else:
                    return self._column_builder(*cres)
            else:
                return cres[0]

    def __setitem__(self, key, val):
        """
           Launches the setitem of the dict found in the block storageobj
           Args:
               key: the position of the value that we want to save
               val: the value that we want to save in that position
           Returns:
        """
        log.debug('SET ITEM %s->%s', key, val)
        if not self._is_persistent:
            dict.__setitem__(self, key, val)
        else:
            self._hcache.put_row(self._make_key(key), self._make_value(val))

    def __repr__(self):
        to_return = {}
        for item in self.iteritems():
            to_return[item[0]] = item[1]
            if len(to_return) == config.hecuba_print_limit:
                return str(to_return)
        if len(to_return) > 0:
            return str(to_return)
        return ""

    def iterkeys(self):
        """
        Obtains the iterator for the keys of the block
        Returns:
            iterkeys(self): list of keys
        """
        if self._is_persistent:
            ik = self._hcache.iterkeys(config.prefetch_size)
            return NamedIterator(ik, self._key_builder, self)
        else:
            return dict.iterkeys(self)

    def iteritems(self):
        """
        Obtains the iterator for the key,val pairs of the block
        Returns:
            BlockItemsIter(self): list of key,val pairs
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
        Obtains the iterator for the values of the block
        Returns:
            BlockValuesIter(self): list of values
        """
        if self._is_persistent:
            ik = self._hcache.itervalues(config.prefetch_size)
            return NamedIterator(ik, self._column_builder, self)
        else:
            return dict.itervalues(self)
