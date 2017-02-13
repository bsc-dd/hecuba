# author: G. Alomar
import uuid
from collections import Iterable
from collections import namedtuple

import logging
from hfetch import Hcache

from IStorage import IStorage
from hecuba import config


class NamedIterator:
    def __init__(self, hiterator, builder):
        self.hiterator = hiterator
        self.builder = builder

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

    args_names = ["keyspace_name", "table_name", "primary_keys", "columns", "is_persistent", "tokens", "storage_id"]
    args = namedtuple('StorageDictArgs', args_names)

    @staticmethod
    def build_remotely(results):
        """
        Launches the Block.__init__ from the api.getByID
        Args:
            results: a list of all information needed to create again the block
        """
        return StorageDict(results.keyspace_name.encode('utf8'),
                           results.table_name.encode('utf8'),
                           results.primary_keys, results.columns, results.tokens,
                           results.storage_id.encode('utf8'))

    def __init__(self, keyspace_name, table_name, primary_keys, columns, is_persistent=True, tokens=None, storage_id=None):
        """
        Creates a new block.

        Args:
            storage_id (string):  an unique block identifier
            peer (string): hostname
            table_name (string): the name of the collection/table
            keyspace_name (string): name of the Cassandra keyspace.
            primary_keys (list(tuple)): a list of (key,type) primary keys (primary + clustering).
            columns (list(tuple)): a list of (key,type) columns
            tokens (list): list of tokens
            storageobj_classname (string): full class name of the storageobj
        """

        if storage_id is None:
            self.storage_id = str(uuid.uuid1())
        else:
            self.storage_id = storage_id

        if tokens is None:
            print 'using all tokens'
            self.tokens = map(lambda a: a.value, config.cluster.metadata.token_map.ring)
        else:
            self.tokens = tokens

        self._build_args = self.args(keyspace_name, table_name, primary_keys, columns, is_persistent,
                                     self.tokens, storage_id)
        self._table = table_name
        self._ksp = keyspace_name
        self.values = columns
        self.is_persistent = is_persistent

        self._primary_keys = primary_keys
        self._columns = columns

        key_names = map(lambda a: a[0], primary_keys)
        column_names = map(lambda a: a[0], columns)
        tknp = "token(%s)" % primary_keys[0]
        self._hcache_params = (config.max_cache_size, keyspace_name, table_name,
                               "WHERE %s>=? AND %s<?;" % (tknp, tknp),
                               tokens, primary_keys, column_names)
        self._item_builder = namedtuple('row', key_names + column_names)
        if self.is_persistent:
            self.hcache = Hcache(*self._hcache_params)
        else:
            self.hcache = None

        if len(key_names) > 1:
            self._key_builder = namedtuple('row', primary_keys)
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
        if not self.is_persistent:
            return dict.__contains__(self, key)
        else:
            try:
                self.hcache.get_row(self._make_key(key))
                return True
            except Exception as e:
                print "Exception in persistentDict.__contains__:", e
                return False

    def _make_key(self, key):
        if isinstance(key, str):
            if len(self._primary_keys) == 1:
                return [key]
            else:
                raise Exception('missing a primary key')

        if isinstance(key, Iterable) and len(key) == len(self._primary_keys):
            return list(key)
        else:
            raise Exception('wrong primary key')

    def _make_value(self, key):
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

    def make_persistent(self):
        self.is_persistent = True
        self.hcache = Hcache(*self._hcache_params)

    def stop_persistent(self):
        self.is_persistent = True
        self.hcache = None

    def delete_persistent(self):
        self.stop_persistent()
        # TODO delete

    def __getitem__(self, key):
        """
        If the object is persistent, each request goes to the prefetcher.
        Args:
             key: the dictionary key
        Returns
             item: value found in position key
        """
        logging.debug('GET ITEM %s', key)

        if not self.is_persistent:
            return dict.__getitem__(self, key)

        else:
            cres = self.hcache.get_row(self._make_key(key))
            if cres is None:
                return None
            else:
                return self._column_builder(cres)

    def __setitem__(self, key, val):
        """
           Launches the setitem of the dict found in the block storageobj
           Args:
               key: the position of the value that we want to save
               val: the value that we want to save in that position
           Returns:
        """
        if not self.is_persistent:
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
        if not self.is_persistent:
            ik = self.hcache.iterkeys()
            return NamedIterator(ik, self._key_builder)
        else:
            return dict.iterkeys(self)

    def iteritems(self):
        """
        Obtains the iterator for the key,val pairs of the block
        Returns:
            BlockItemsIter(self): list of key,val pairs
        """
        if not self.is_persistent:
            ik = self.hcache.iteritems()
            return NamedItemsIterator(self._key_builder, self._column_builder, self._k_size, ik)
        else:
            return dict.iteritems(self)

    def itervalues(self):
        """
        Obtains the iterator for the values of the block
        Returns:
            BlockValuesIter(self): list of values
        """
        if not self.is_persistent:
            ik = self.hcache.itervalues()
            return NamedIterator(ik, self._column_builder)
        else:
            return dict.itervalues(self)
