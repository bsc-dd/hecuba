from IStorage import IStorage, AlreadyPersistentError
from hecuba import config, log
from collections import namedtuple
import re
import uuid


class StorageSet(set, IStorage):
    args_names = ["name", "column", "tokens", "storage_id", "istorage_props", "class_name"]
    args = namedtuple('StorageSetArgs', args_names)
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba' +
                                                  '.istorage (storage_id, class_name, name, tokens, istorage_props) '
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
        log.debug("StorageSet: storing media %s", storage_args)
        try:

            config.session.execute(StorageSet._prepared_store_meta,
                                   [storage_args.storage_id,
                                    storage_args.class_name,
                                    storage_args.name,
                                    storage_args.tokens,
                                    storage_args.istorage_props])
        except Exception as ex:
            log.warn("Error creating the StorageSet metadata: %s, %s", str(storage_args), ex)
            raise ex

    _set_case = re.compile('.*@TypeSpec +(\w+)')
    _tuple_case = re.compile('.*@TypeSpec *< *([\w, +]+) *>')

    def __init__(self, name="", column=None, tokens=None, storage_id=None, istorage_props=None, **kwargs):
        """
            Creates a new storageset.
            Args:
                name (string): the name of the Cassandra Keyspace + table where information can be found
                tokens (list of tuples): token ranges assigned to the new StorageSet
                storage_id (string):  an unique storageset identifier
                istorage_props dict(string,string): a map with the storage id of each contained istorage object.
                kwargs: more optional parameters
        """
        super(StorageSet, self).__init__(**kwargs)
        log.debug("CREATED StorageSet(%s)", name)
        self._is_persistent = False
        self._storage_id = storage_id
        self._istorage_props = istorage_props
        self._tokens = tokens
        self._ksp, self._table = self._extract_ks_tab(name)
        self._class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)

        if tokens is None:
            log.info('using all tokens')
            tokens = map(lambda a: a.value, config.cluster.metadata.token_map.ring)
            self._tokens = IStorage._discrete_token_ranges(tokens)
        else:
            self._tokens = tokens

        if self.__doc__ is not None:
            self._persistent_props = self._parse_comments(self.__doc__)
            self._persistent_attrs = self._persistent_props.keys()
            self._column = self._persistent_props['column']
        else:
            self._column = column

        class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)
        self._build_args = self.args(None, self._column, self._tokens,
                                     self._storage_id, self._istorage_props, class_name)

        if name:
            self._setup_persistent_structs()
            self._store_meta(self._build_args)
        else:
            self._is_persistent = False

    @classmethod
    def _parse_comments(cls, comments):
        """
            Parses de comments in a class file to save them in the class information
            Args:
                comments: the comment in the class file
            Returns:
                this: a structure with all the information of the comment
        """
        this = {}
        for line in comments.split('\n'):
            m = StorageSet._set_case.match(line)
            if m is not None:
                # Matching TypeSpec of a Set
                set_types = m.groups()
                set_type = set_types[0]

                set_type_cassandra = StorageSet._conversions[set_type]

                if cls.__class__.__name__ in this:
                    this.update({'type': 'set', 'column': set_type_cassandra})
                else:
                    this = {
                        'type': 'set',
                        'column': set_type_cassandra}
            else:
                # Matching TypeSpec of a Set of Tuples
                m = StorageSet._tuple_case.match(line)
                if m is not None:
                    types = m.groups()[0]
                    simple_type = types.replace(' ', '')
                    simple_type_split = simple_type.split(',')
                    conversion = list()
                    for ind, val in enumerate(simple_type_split):
                        if ind == 0:
                            conversion.append(IStorage._conversions[val])
                        else:
                            conversion.append(IStorage._conversions[val])
                    this = {
                        'type': 'tuple',
                        'column': conversion
                    }

        return this

    def make_persistent(self, name):
        """
        Method to transform a StorageSet into a persistent object.
        This will make it use a persistent DB as the main location
        of its data.
        Args:
            name (string): name with which the table in the DB will be created
        """

        if self._is_persistent:
            raise AlreadyPersistentError("This StorageSet is already persistent [Before:{}.{}][After:{}]",
                                         self._ksp, self._table, name)

        (self._ksp, self._table) = self._extract_ks_tab(name)
        self._build_args = self._build_args._replace(name=self._ksp + '.' + self._table)
        self._setup_persistent_structs()

        ps = config.session.prepare(
            "INSERT INTO %s.%s (column, empty_column) VALUES (?, ' ')" % (self._ksp, self._table))
        for value in set(self):
            config.session.execute(ps, [value])

        self._store_meta(self._build_args)

    def add(self, value):
        """
           Method to insert values in the StorageSet
           Args:
               value: the value that we want to save
        """

        if self._is_persistent:
            query = "INSERT INTO %s.%s (column, empty_column)" % (self._ksp, self._table)
            if isinstance(value, str) or isinstance(value, unicode):
                query += " VALUES ('%s', ' ')" % value
            elif isinstance(value, tuple):
                query += " VALUES (("
                for val in value:
                    if isinstance(val, unicode):
                        val = str(val)
                    if isinstance(val, str):
                        query += "'" + val + "', "
                    else:
                        query += str(val) + ", "

                query = query[:-2] + "), ' ')"
            else:
                query += " VALUES (%s, ' ')" % value
            config.session.execute(query)
        else:
            set.add(self, value)

    def remove(self, value):
        """
           Method to delete values in the StorageSet
           Args:
               value: the value that we want to delete
        """
        if self._is_persistent:
            query = "DELETE FROM %s.%s WHERE column = " % (self._ksp, self._table)
            if isinstance(value, str) or isinstance(value, unicode):
                query += "'%s'" % value
            elif isinstance(value, tuple):
                query += "("
                for val in value:
                    if isinstance(val, unicode):
                        val = str(val)
                    if isinstance(val, str):
                        query += "'" + val + "', "
                    else:
                        query += str(val) + ", "

                query = query[:-2] + ")"
            else:
                query += str(value)
            config.session.execute(query)
        else:
            set.remove(self, value)

    def __contains__(self, value):
        """
           Method to check if a given value is in the StorageSet
           Args:
               value: the value that we want to check
        """

        if self._is_persistent:
            query = "SELECT count(*) FROM %s.%s WHERE column = " % (self._ksp, self._table)
            if isinstance(value, str) or isinstance(value, unicode):
                query += "'%s'" % value
            elif isinstance(value, tuple):
                query += "("
                for val in value:
                    if isinstance(val, unicode):
                        val = str(val)
                    if isinstance(val, str):
                        query += "'" + val + "', "
                    else:
                        query += str(val) + ", "

                query = query[:-2] + ")"
            else:
                query += str(value)
            result = config.session.execute(query)
            # result[0] is the first row (will be only on row) and result[0][0] is the count
            return result[0][0]
        else:
            return set.__contains__(self, value)

    def union(self, set2):
        for value in set2:
            self.add(value)

        return self

    def intersection(self, set2):
        for value in self:
            if value not in set2:
                self.remove(value)

        return self

    def difference(self, set2):
        if len(self) <= len(set2):
            for value in self:
                if value in set2:
                    self.remove(value)
        else:
            for value in set2:
                if value in self:
                    self.remove(value)

        return self

    def clear(self):
        if self._is_persistent:
            query = "TRUNCATE %s.%s" % (self._ksp, self._table)
            config.session.execute(query)
        else:
            set.clear(self)

    def __iter__(self):
        if self._is_persistent:
            query = "SELECT column FROM %s.%s " % (self._ksp, self._table)
            result = config.session.execute(query)
            result = map(lambda x: x[0], result)
            return iter(result)
        else:
            return set.__iter__(self)

    def __len__(self):
        if self._is_persistent:
            query = "SELECT count(*) FROM %s.%s " % (self._ksp, self._table)
            result = config.session.execute(query)
            return result[0][0]
        else:
            return len(set(self))

    def _setup_persistent_structs(self):
        """
            Setups the python structures used to communicate with the backend.
            Creates the necessary tables on the backend to store the object data.
        """

        self._is_persistent = True

        if self._storage_id is None:
            self._storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, self._ksp + '.' + self._table)

        self._build_args = self._build_args._replace(storage_id=self._storage_id)
        self._store_meta(self._build_args)

        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (self._ksp, config.replication)
        try:
            config.session.execute(query_keyspace)
        except Exception as ex:
            log.warn("Error creating the StorageSet keyspace %s, %s", (query_keyspace), ex)
            raise ex

        query_simple = 'CREATE TABLE IF NOT EXISTS ' + self._ksp + '.' + self._table + \
                       '( '

        query_simple += "column "
        if self._persistent_props['type'] == 'tuple':
            query_simple += "tuple<"
            for num, type in enumerate(self._persistent_props['column']):
                if num != 0:
                    query_simple += ", "
                query_simple += type
            query_simple += ">"
        else:
            query_simple += self._persistent_props['column']
        query_simple += " PRIMARY KEY, empty_column text, "
        try:
            config.session.execute(query_simple[:-2] + ' )')
        except Exception as ir:
            log.error("Unable to execute %s", query_simple)
            raise ir

    def stop_persistent(self):
        """
            The StorageSet stops being persistent, but keeps the information already stored in Cassandra
        """
        if not self._is_persistent:
            raise Exception("This StorageSet is not persistent.")
        log.debug("STOP PERSISTENT")
        self._is_persistent = False
        # We have to update the set in memory
        query = "SELECT column FROM %s.%s " % (self._ksp, self._table)
        result = config.session.execute(query)
        result = map(lambda x: x[0], result)
        set.clear(self)
        for value in result:
            self.add(value)

    def delete_persistent(self):
        """
            Deletes the Cassandra table where the persistent StorageSet stores data
        """
        if not self._is_persistent:
            raise Exception("This StorageSet is not persistent.")

        self._is_persistent = False
        # We have to update the set in memory
        query = "SELECT column FROM %s.%s " % (self._ksp, self._table)
        result = config.session.execute(query)
        result = map(lambda x: x[0], result)
        set.clear(self)
        for value in result:
            self.add(value)

        query = "DROP TABLE IF EXISTS %s.%s;" % (self._ksp, self._table)
        log.debug("DELETE PERSISTENT: %s", query)
        config.session.execute(query)

