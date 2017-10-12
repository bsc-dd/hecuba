from IStorage import IStorage, AlreadyPersistentError
from hecuba import config, log

from hfetch import Hcache

import uuid
from collections import namedtuple
import numpy as np


class StorageNumpy(np.ndarray, IStorage):
    _storage_id = None
    _build_args = None
    _class_name = None
    _hcache_params = None
    _hcache = None
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba.istorage'
                                                  '(storage_id, class_name, name)'
                                                  'VALUES (?,?,?)')

    args_names = ["storage_id", "class_name", "name"]
    args = namedtuple('StorageNumpyArgs', args_names)

    def __new__(cls, input_array=None, storage_id=None, name=None, **kwargs):

        if input_array is None and name is not None and storage_id is not None:
            input_array = cls.load_array(storage_id, name)
            obj = np.asarray(input_array).view(cls)
            obj._is_persistent = True
        elif name is None and storage_id is not None:
            raise RuntimeError("hnumpy received storage id but not a name")
        elif (input_array is not None and name is not None and storage_id is not None) \
                or (storage_id is None and name is not None):
            obj = np.asarray(input_array).view(cls)
            obj.make_persistent(name)
        else:
            obj = np.asarray(input_array).view(cls)
            obj._is_persistent = False
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        # add the new attribute to the created instance
        obj._storage_id = storage_id
        # Finally, we must return the newly created object:
        obj._class_name = '%s.%s' % (cls.__module__, cls.__name__)
        return obj

    # used as copy constructor
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._storage_id = getattr(obj, '_storage_id', None)

    @staticmethod
    def build_remotely(new_args):
        """
            Launches the StorageNumpy.__init__ from the uuid api.getByID
            Args:
                new_args: a list of all information needed to create again the StorageNumpy
            Returns:
                so: the created StorageNumpy
        """
        log.debug("Building StorageNumpy object with %s", new_args)
        return StorageNumpy(new_args.storage_id)

    @staticmethod
    def _store_meta(storage_args):
        """
            Saves the information of the object in the istorage table.
            Args:.
                storage_args (object): contains all data needed to restore the object from the workers
        """
        log.debug("StorageObj: storing media %s", storage_args)
        try:
            config.session.execute(StorageNumpy._prepared_store_meta,
                                   [storage_args.storage_id, storage_args.class_name,
                                    storage_args.name])
        except Exception as ex:
            log.warn("Error creating the StorageNumpy metadata with args: %s" % str(storage_args))
            raise ex

    @staticmethod
    def load_array(storage_id, name):
        (ksp, table) = IStorage._extract_ks_tab(name)
        _hcache_params = (ksp, table + '_numpies',
                          storage_id, [], ['storage_id', 'cluster_id', 'block_id'],
                          [{'name': "payload", 'type': 'numpy'}],
                          {'cache_size': config.max_cache_size,
                           'writer_par': config.write_callbacks_number,
                           'write_buffer': config.write_buffer_size})
        _hcache = Hcache(*_hcache_params)
        result = _hcache.get_row([storage_id, -1, -1])
        if len(result) == 1:
            return result[0]
        else:
            raise KeyError

    def make_persistent(self, name):
        if self._is_persistent:
            raise AlreadyPersistentError("This StorageNumpy is already persistent [Before:{}.{}][After:{}]",
                                         self._ksp, self._table, name)
        self._is_persistent = True

        (self._ksp, self._table) = self._extract_ks_tab(name)
        if self._storage_id is None:
            self._storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, self._ksp + '.' + self._table + '_numpies')
        self._build_args = self.args(self._storage_id, self._class_name, name)
        log.info("PERSISTING DATA INTO %s %s", self._ksp, self._table)

        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s }" % (self._ksp, config.replication)
        config.session.execute(query_keyspace)

        config.session.execute('CREATE TABLE IF NOT EXISTS ' + self._ksp + '.' + self._table + '_numpies'
                                                                                               '(storage_id uuid , '
                                                                                               'cluster_id int, '
                                                                                               'block_id int, '
                                                                                               'payload blob, '
                                                                                               'PRIMARY KEY((storage_id,cluster_id),block_id))')

        self._hcache_params = (self._ksp, self._table + '_numpies',
                               self._storage_id, [], ['storage_id', 'cluster_id', 'block_id'],
                               [{'name': "payload", 'type': 'numpy'}],
                               {'cache_size': config.max_cache_size,
                                'writer_par': config.write_callbacks_number,
                                'write_buffer': config.write_buffer_size})

        self._hcache = Hcache(*self._hcache_params)
        if len(self.shape) != 0:
            self._hcache.put_row([self._storage_id, -1, -1], [self])
        self._store_meta(self._build_args)

    def delete_persistent(self):
        """
            Deletes the Cassandra table where the persistent StorageObj stores data
        """
        self._is_persistent = False

        query = "DELETE FROM %s.%s WHERE storage_id = %s;" % (self._ksp, self._table + '_numpies', self._storage_id)
        log.debug("DELETE PERSISTENT: %s", query)
        config.session.execute(query)

        # TODO should I also drop the table when empty?
        # TODO DELETE THE METAS
        ##to overload [] override __set_item__ and __get_item__
