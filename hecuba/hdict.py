# author: G. Alomar
import uuid
from collections import Iterable
from collections import namedtuple

import logging
from types import NoneType

from hfetch import Hcache

from IStorage import IStorage
from hecuba import config


class NamedIterator:
    def __init__(self, hiterator, builder):
        self.hiterator = hiterator
        self.builder = builder

    def __iter__(self):
        return self

    def next(self):
        n = self.hiterator.get_next()
        if n is None:
            raise StopIteration
        elif self.builder is not None:
            return self.builder(n)
        else:
            return n[0]


class NamedItemsIterator:
    builder = namedtuple('row', 'key, value')

    def __init__(self, key_builder, column_builder, k_size, hiterator):
        self.key_builder = key_builder
        self.k_size = k_size
        self.column_builder = column_builder
        self.hiterator = hiterator

    def __iter__(self):
        return self

    def next(self):
        n = self.hiterator.get_next()
        if n is None:
            raise StopIteration
        else:
            if self.key_builder is None:
                k = n[0]
            else:
                k = self.key_builder(*n[0:self.k_size])
            if self.column_builder is None:
                v = n[self.k_size]
            else:
                v = self.column_builder(*n[self.k_size:])
            return self.builder(k, v)


class StorageDict(dict, IStorage):
    """
    Object used to access data from workers.
    """

    args_names = ["primary_keys", "columns", "name", "tokens","storage_id"]
    args = namedtuple('StorageDictArgs', args_names)
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba.istorage (storage_id, class_name,'
                                       ' name, tokens,dict_pks,dict_columns)  VALUES (?,?,?,?,?,?)')

    @staticmethod
    def build_remotely(storage_id):
        """
        Launches the Block.__init__ from the api.getByID
        Args:
            storage_id: a list of all information needed to create again the block
        """

        return StorageDict(storage_id.primary_keys,
                           storage_id.columns,
                           storage_id.name,
                           storage_id.tokens
                           )
    @staticmethod
    def _store_meta(storage_args):
        class_name = '%s.%s' % (StorageDict.__class__.__module__, StorageDict.__class__.__name__)

        try:
            config.session.execute(StorageDict._prepared_store_meta,
                                   [storage_args.storage_id, class_name, storage_args.name,
                                    storage_args.tokens, storage_args.primary_keys, storage_args.columns])
        except Exception as ex:
            print "Error creating the StorageDict metadata:", storage_args, ex
            raise ex

    def __init__(self, primary_keys, columns, name=None, tokens=None):
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

        if tokens is None:
            print 'using all tokens'
            tokens = map(lambda a: a.value, config.cluster.metadata.token_map.ring)
            tokens.sort()
            self.tokens = []
            n_tns = len(tokens)
            for i in range(0, n_tns):
                self.tokens.append((tokens[i], tokens[(i + 1) % n_tns]))
        else:
            self.tokens = tokens

        self._build_args = self.args(primary_keys, columns, name, self.tokens, None)
        self._primary_keys = primary_keys
        self._columns = columns

        self.values = columns
        if name is not None:
            self._is_persistent = True
            self.make_persistent(name)
        else:
            self._is_persistent = False
        key_names = map(lambda a: a[0], self._primary_keys)
        column_names = map(lambda a: a[0], self._columns)
        self._item_builder = namedtuple('row', map(lambda a: a[0], primary_keys + columns))

        if len(key_names) > 1:
            self._key_builder = namedtuple('row', key_names)
        else:
            self._key_builder = None
        if len(column_names) > 1:
            self._column_builder = namedtuple('row', column_names)
        else:
            self._column_builder = None
        self._k_size = len(key_names)

    def __eq__(self, other):
        return self.dict_id == other.blockid and \
               self.tokens == other.token_ranges \
               and self._table == other.table_name and self._ksp == other.keyspace

    def __contains__(self, key):
        if not self._is_persistent:
            return dict.__contains__(self, key)
        else:
            try:
                self.hcache.get_row(self._make_key(key))
                return True
            except Exception as e:
                print "Exception in persistentDict.__contains__:", e
                return False

    def _make_key(self, key):
        if isinstance(key, str) or not isinstance(key, Iterable):
            if len(self._primary_keys) == 1:
                return [key]
            else:
                raise Exception('missing a primary key')

        if isinstance(key, Iterable) and len(key) == len(self._primary_keys):
            return list(key)
        else:
            raise Exception('wrong primary key')

    @staticmethod
    def _make_value(key):
        if isinstance(key, str) or not isinstance(key, Iterable):
            return [key]
        else:
            return list(key)

    def __iadd__(self, key, other):
        """
        Implements the += logic.
        This method is consistent only if called on a counter.

        Args:
             key : the key to update
             other: value to add
        Returns:
            None
        """

        if self.is_counter:
            self[key] = other
        else:
            self[key] = self[key] + other

    def keys(self):
        """
        This method return a list of all the keys of the PersistentDict.
        Returns:
          list: a list of keys
        """
        return [i for i in self.iterkeys()]

    def __iter__(self):
        return self.iterkeys()

    def make_persistent(self, name):
        (self._ksp, self._table) = self._extract_ks_tab(name)

        if not hasattr(self, 'storage_id'):
            self.storage_id = str(uuid.uuid1())
            self._build_args = self._build_args._replace(storage_id=self.storage_id)

        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = {'class': 'SimpleStrategy'," \
                         "'replication_factor': %d }" % (self._ksp, config.repl_factor)
        try:
            config.session.execute(query_keyspace)
        except Exception as ex:
            print "Error creating the StorageDict keyspace:", query_keyspace, ex
            raise ex

        columns = map(lambda a: a[0] + " " + a[1], self._primary_keys + self._columns)
        pks = map(lambda a: a[0], self._primary_keys)
        query_table = "CREATE TABLE IF NOT EXISTS %s.%s (%s, PRIMARY KEY (%s));" % (self._ksp, self._table,
                                                                                    str.join(',', columns),
                                                                                    str.join(',', pks))
        try:
            config.session.execute(query_table)
        except Exception as ex:
            print "Error creating the StorageDict table:", query_table, ex
            raise ex

        self._store_meta(self._build_args)
        key_names = map(lambda a: a[0], self._primary_keys)
        column_names = map(lambda a: a[0], self._columns)
        tknp = "token(%s)" % key_names[0]
        self._hcache_params = (config.max_cache_size, self._ksp, self._table,
                               "WHERE %s>=? AND %s<?;" % (tknp, tknp),
                               self.tokens, key_names, column_names)
        print "HCACHE paramets", self._hcache_params
        self.hcache = Hcache(*self._hcache_params)
        # Storing all in-memory values to cassandra
        for key, value in self.iteritems():
            self.hcache.put_row(self._make_key(key), self._make_value(value))
        self._is_persistent = True

    def stop_persistent(self):
        self._is_persistent = True
        self.hcache = None

    def delete_persistent(self):
        self.stop_persistent()
        query = "DROP TABLE IF EXISTS %s.%s;" % (self._ksp, self._table)
        config.session.execute(query)

    def __getitem__(self, key):
        """
        If the object is persistent, each request goes to the prefetcher.
        Args:
             key: the dictionary key
        Returns
             item: value found in position key
        """
        logging.debug('GET ITEM %s', key)

        if not self._is_persistent:
            return dict.__getitem__(self, key)

        else:
            cres = self.hcache.get_row(self._make_key(key))
            print cres, "-->", cres.__class__

            if issubclass(cres.__class__, NoneType):
                return None
            elif self._column_builder is not None:
                return self._column_builder(cres)
            else:
                return cres

    def __setitem__(self, key, val):
        """
           Launches the setitem of the dict found in the block storageobj
           Args:
               key: the position of the value that we want to save
               val: the value that we want to save in that position
           Returns:
        """
        if not self._is_persistent:
            return dict.__setitem__(self, key, val)
        else:
            return self.hcache.put_row(self._make_key(key) + self._make_value(val))

    def getID(self):
        """
        Obtains the id of the block
        Returns:
            self.blockid: id of the block
        """
        return self.dict_id

    def iterkeys(self):
        """
        Obtains the iterator for the keys of the block
        Returns:
            iterkeys(self): list of keys
        """
        if self._is_persistent:
            ik = self.hcache.iterkeys(config.prefetch_size)
            return NamedIterator(ik, self._key_builder)
        else:
            return dict.iterkeys(self)

    def iteritems(self):
        """
        Obtains the iterator for the key,val pairs of the block
        Returns:
            BlockItemsIter(self): list of key,val pairs
        """
        if self._is_persistent:
            ik = self.hcache.iteritems(config.prefetch_size)
            return NamedItemsIterator(self._key_builder, self._column_builder, self._k_size, ik)
        else:
            return dict.iteritems(self)

    def itervalues(self):
        """
        Obtains the iterator for the values of the block
        Returns:
            BlockValuesIter(self): list of values
        """
        if self._is_persistent:
            ik = self.hcache.itervalues(config.prefetch_size)
            return NamedIterator(ik, self._column_builder)
        else:
            return dict.itervalues(self)
