import re
from copy import copy
from uuid import uuid1

from IStorage import IStorage
from hdict import StorageDict
from hecuba import config, log


class StorageObj(object, IStorage):
    """
    This class is where information will be stored in Hecuba.
    The information can be in memory, stored in a python dictionary or local variables, or saved in a
    DDBB(Cassandra), depending on if it's persistent or not.
    """

    @staticmethod
    def build_remotely(storage_id):
        """
        Launches the StorageObj.__init__ from the api.getByID
        Args:
            storage_id: a list of all information needed to create again the storageobj
        Returns:
            so: the created storageobj
        """
        class_name = storage_id.class_name
        if class_name is 'StorageObj':
            so = StorageObj(storage_id.name.encode('utf8'))

        else:
            last = 0
            for key, i in enumerate(class_name):
                if i == '.' and key > last:
                    last = key
            module = class_name[:last]
            cname = class_name[last + 1:]
            mod = __import__(module, globals(), locals(), [cname], 0)
            so = getattr(mod, cname)(storage_id.name.encode('utf8'))

        return so

    @staticmethod
    def _store_meta(storage_args):
        class_name = '%s.%s' % (StorageDict.__class__.__module__, StorageDict.__class__.__name__)

        try:
            config.session.execute('INSERT INTO hecuba.istorage (storage_id, class_name, name)  VALUES (%s,%s,%s)',
                                   [storage_args.storage_id, class_name, storage_args.name])
        except Exception as ex:
            print "Error creating the StorageDict metadata:", storage_args, ex
            raise ex

    def __init__(self, name=None):
        """
        Creates a new storageobj.
        Args:
            name (string): the name of the Cassandra Keyspace + table where information can be found
            storage_id (string):  an unique storageobj identifier
        """
        log.debug("CREATED StorageObj(%s)", name)
        self._persistent_dicts = []
        self._attr_to_column = {}

        self._persistent_props = self._parse_comments(self.__doc__)

        self._is_persistent = False
        dictionaries = filter(lambda (k, t): t['type'] == 'dict', self._persistent_props.iteritems())
        for table_name, per_dict in dictionaries:
            pd = StorageDict(per_dict['primary_keys'], per_dict['columns'])
            setattr(self, table_name, pd)
            self._persistent_dicts.append(pd)

        if name is not None:
            self.make_persistent(name)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.getID() == other.getID()

    _valid_type = '(atomicint|str|bool|decimal|float|int|tuple|list|generator|frozenset|set|dict|long|buffer|' \
                  'bytearray|counter)'
    _data_type = re.compile('(\w+) *: *%s' % _valid_type)
    _dict_case = re.compile('.*@ClassField +(\w+) +dict +< *< *([\w:, ]+)+ *> *, *([\w+:, <>]+) *>')
    _sub_dict_case = re.compile(' *< *< *([\w:, ]+)+ *> *, *([\w+:, <>]+) *>')
    _sub_tuple_case = re.compile(' *< *([\w:, ]+)+ *>')
    _val_case = re.compile('.*@ClassField +(\w+) +(\w+) +%s' % _valid_type)
    _index_vars = re.compile('.*@Index_on+\s*([A-z,]+)+([A-z, ]+)')

    @staticmethod
    def _parse_comments(comments):
        """
        Parses de comments in a class file to save them in the class information
        Args:
           comments: the comment in the class file
        """
        this = {}
        for line in comments.split('\n'):
            m = StorageObj._dict_case.match(line)
            if m is not None:
                # Matching @ClassField of a dict
                table_name, dict_keys, dict_vals = m.groups()
                primary_keys = []
                for ind, key in enumerate(dict_keys.split(",")):
                    try:
                        name, value = StorageObj._data_type.match(key).groups()
                    except Exception:
                        if ':' in key:
                            raise SyntaxError
                        else:
                            name = "key" + str(ind)
                            value = key
                    name = name.replace(' ', '')
                    primary_keys.append((name, StorageObj._conversions[value]))
                dict_vals = dict_vals.replace(' ', '')
                if dict_vals.startswith('dict'):
                    n = StorageObj._sub_dict_case.match(dict_vals[4:])
                    dict_keys2, dict_vals2 = n.groups()
                    primary_keys2 = []
                    for ind, key in enumerate(dict_keys2.split(",")):
                        try:
                            name, value = StorageObj._data_type.match(key).groups()
                        except ValueError:
                            if ':' in key:
                                raise SyntaxError
                            else:
                                name = "key" + str(ind)
                                value = key
                        name = name.replace(' ', '')
                        primary_keys2.append((name, StorageObj._conversions[value]))
                    columns2 = []
                    dict_vals2 = dict_vals2.replace(' ', '')
                    if dict_vals2.startswith('tuple'):
                        dict_vals2 = dict_vals2[6:]
                    for ind, val in enumerate(dict_vals2.split(",")):
                        try:
                            name, value = StorageObj._data_type.match(val).groups()
                        except ValueError:
                            if ':' in key:
                                raise SyntaxError
                            else:
                                name = "val" + str(ind)
                                value = val
                        columns2.append((name, StorageObj._conversions[value]))
                    columns = {
                        'type': 'dict',
                        'primary_keys': primary_keys2,
                        'columns': columns2}
                elif dict_vals.startswith('tuple'):
                    n = StorageObj._sub_tuple_case.match(dict_vals[5:])
                    tuple_vals = list(n.groups())[0]
                    columns = []
                    for ind, val in enumerate(tuple_vals.split(",")):
                        try:
                            name, value = val.split(':')
                        except ValueError:
                            if ':' in key:
                                raise SyntaxError
                            else:
                                name = "val" + str(ind)
                                value = val
                        name = name.replace(' ', '')
                        columns.append((name, StorageObj._conversions[value]))
                else:
                    columns = []
                    for ind, val in enumerate(dict_vals.split(",")):
                        try:
                            name, value = StorageObj._data_type.match(val).groups()
                        except ValueError:
                            if ':' in key:
                                raise SyntaxError
                            else:
                                name = "val" + str(ind)
                                value = val
                        name = name.replace(' ', '')
                        columns.append((name, StorageObj._conversions[value]))
                if table_name in this:
                    this[table_name].update({'type': 'dict', 'primary_keys': primary_keys, 'columns': columns})
                else:
                    this[table_name] = {
                        'type': 'dict',
                        'primary_keys': primary_keys,
                        'columns': columns}
            else:
                m = StorageObj._val_case.match(line)
                if m is not None:
                    table_name, simple_type = m.groups()
                    this[table_name] = {
                        'type': StorageObj._conversions[simple_type]
                    }
            m = StorageObj._index_vars.match(line)
            if m is not None:
                table_name, indexed_values = m.groups()
                indexed_values = indexed_values.replace(' ', '').split(',')
                if table_name in this:
                    this[table_name].update({'indexed_values': indexed_values})
                else:
                    this[table_name] = {'indexed_values': indexed_values}
        return this

    def make_persistent(self, name):
        """
        Once a StorageObj has been created, it can be made persistent. This function retrieves the information about
        the Object class schema, and creates a Cassandra table with those parameters, where information will be saved
        from now on, until execution finishes or StorageObj is no longer persistent.
        It also inserts into the new table all information that was in memory assigned to the StorageObj prior to this
        call.
        """
        (self._ksp, self._table) = self._extract_ks_tab(name)
        log.info("PERSISTING DATA INTO %s %s", self._ksp, self._table)

        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = {'class': 'SimpleStrategy'," \
                         "'replication_factor': %d }" % (self._ksp, config.repl_factor)
        try:
            config.session.execute(query_keyspace)
        except Exception as ex:
            print "Error executing query:", query_keyspace
            raise ex

        dictionaries = filter(lambda (k, t): t['type'] == 'dict', self._persistent_props.iteritems())
        for table_name, _ in dictionaries:
            getattr(self, table_name).make_persistent(self._ksp + "." + table_name)

        create_attrs = "CREATE TABLE IF NOT EXISTS %s.%s (" % (self._ksp, self._table) + \
                       "name text PRIMARY KEY, " + \
                       "intval int, " + \
                       "intlist list<int>, " + \
                       "inttuple list<int>, " + \
                       "doubleval double, " + \
                       "doublelist list<double>, " + \
                       "doubletuple list<double>, " + \
                       "textval text, " + \
                       "textlist list<text>, " + \
                       "texttuple list<text>)"
        config.session.execute(create_attrs)

        self._is_persistent = True

        for key, variable in vars(self).iteritems():
            if not key[0] == '_':
                insert_query = 'INSERT INTO %s.%s (name, ' % (self._ksp, self._table)
                if isinstance(variable, list):
                    if isinstance(variable[0], int):
                        insert_query += 'intlist) VALUES (\'' + str(key) + '\', ' + str(variable) + ')'
                        config.session.execute(insert_query)
                    if isinstance(variable[0], float):
                        insert_query += 'doubleval) VALUES (\'' + str(key) + '\', ' + str(variable) + ')'
                        config.session.execute(insert_query)
                    if isinstance(variable[0], str):
                        insert_query += 'textlist) VALUES (\'' + str(key) + '\', ' + str(variable) + ')'
                        config.session.execute(insert_query)
                elif isinstance(variable, tuple):
                    if isinstance(variable[0], int):
                        insert_query += 'inttuple) VALUES (\'' + str(key) + '\', ' + str(list(variable)) + ')'
                        config.session.execute(insert_query)
                    if isinstance(variable[0], float):
                        insert_query += 'doubletuple) VALUES (\'' + str(key) + '\', ' + str(list(variable)) + ')'
                        config.session.execute(insert_query)
                    if isinstance(variable[0], str):
                        insert_query += 'texttuple) VALUES (\'' + str(key) + '\', ' + str(list(variable)) + ')'
                        config.session.execute(insert_query)
                else:
                    if isinstance(variable, int):
                        insert_query += 'intval) VALUES (\'' + str(key) + '\', ' + str(variable) + ')'
                        config.session.execute(insert_query)
                    if isinstance(variable, float):
                        insert_query += 'doubleval) VALUES (\'' + str(key) + '\', ' + str(variable) + ')'
                        config.session.execute(insert_query)
                    if isinstance(variable, str):
                        insert_query += 'textval) VALUES (\'' + str(key) + '\', \'' + str(variable) + '\')'
                        config.session.execute(insert_query)

    def stop_persistent(self):
        """
            Empties the Cassandra table where the persistent StorageObj stores data
        """

        log.debug("STOP PERSISTENT")
        for pdict in self._persistent_dicts:
            pdict.stop_persistent()

    def delete_persistent(self):
        """
            Deletes the Cassandra table where the persistent StorageObj stores data
        """
        self._is_persistent = False
        for pdict in self._persistent_dicts:
            pdict.delete_persistent()

        query = "TRUNCATE TABLE %s.%s;" % (self._ksp, self._table)
        log.debug("DELETE PERSISTENT: %s", query)
        config.session.execute(query)

    def __getattr__(self, key):
        """
        Given a key, this function reads the configuration table in order to know if the attribute can be found:
        a) In memory
        b) In the DDBB
        Args:
            key: name of the value that we want to obtain
        Returns:
            value: obtained value
        """

        if key[0] != '_' and self._is_persistent:
            try:
                query = "SELECT * FROM %s.%s WHERE name = '%s';" % (self._ksp, self._table, key)
                log.debug("GETATTR: ", query)
                result = config.session.execute(query)
                for row in result:
                    for rowkey, rowvar in vars(row).iteritems():
                        if not rowkey == 'name':
                            if rowvar is not None:
                                if rowkey == 'doubletuple' or rowkey == 'inttuple' or rowkey == 'texttuple':
                                    return tuple(rowvar)
                                else:
                                    return rowvar
            except Exception as ex:
                log.warn("GETATTR ex %s", ex)
                col_name = self._attr_to_column[key]
                query = "SELECT %s FROM %s.%s" % (col_name, self._ksp, self._table) + "WHERE name = %s"
                log.debug("GETATTR2: ", query)
                result = config.session.execute(query, [key])
                if len(result) == 0:
                    raise KeyError('value not found')
                val = getattr(result[0], col_name)
                if 'tuple' in col_name:
                    val = tuple(val)
                return val
        else:
            return super(StorageObj, self).__getattribute__(key)

    def __setattr__(self, key, value):
        """
        Given a key and its value, this function saves it (depending on if it's persistent or not):
            a) In memory
            b) In the DDBB
        Args:
            key: name of the value that we want to obtain
            value: value that we want to save
        """
        if issubclass(value.__class__, IStorage):
            super(StorageObj, self).__setattr__(key, value)
        elif key[0] is '_' or key is 'storage_id':
            object.__setattr__(self, key, value)

        elif self._is_persistent:
            if type(value) == int:
                self._attr_to_column[key] = 'intval'
            if type(value) == str:
                self._attr_to_column[key] = 'textval'
            if type(value) == list or type(value) == tuple:
                if len(value) == 0:
                    return
                first = value[0]
                if type(first) == int:
                    self._attr_to_column[key] = 'intlist'
                elif type(first) == str:
                    self._attr_to_column[key] = 'textlist'
                elif type(first) == float:
                    self._attr_to_column[key] = 'doublelist'
            if type(value) == tuple:
                if len(value) == 0:
                    return
                first = value[0]
                if type(first) == int:
                    self._attr_to_column[key] = 'inttuple'
                elif type(first) == str:
                    self._attr_to_column[key] = 'texttuple'
                elif type(first) == float:
                    self._attr_to_column[key] = 'doubletuple'
                value = list(value)

            query = "INSERT INTO %s.%s (name,%s)" % (self._ksp, self._table, self._attr_to_column[key])
            query += "VALUES (%s,%s)"
            log.debug("SETATTR: ", query)
            config.session.execute(query, [key, value])
        else:
            super(StorageObj, self).__setattr__(key, value)

    def getID(self):
        """
        This function returns the ID of the StorageObj
        """
        if self.storage_id is None:
            self.storage_id = uuid1()
        return '%s_1' % self.storage_id

    def split(self):
        """
          Depending on if it's persistent or not, this function returns the list of keys of the PersistentDict
          assigned to the StorageObj, or the list of keys
          Returns:
               a) List of keys in case that the SO is not persistent
               b) Iterator that will return Blocks, one by one, where we can find the SO data in case it's persistent
        """
        log.debug("SPLITTING StorageObj")
        props = self._persistent_props
        dictionaries = filter(lambda (k, t): t['type'] == 'dict', props.iteritems())
        dicts = []
        for table_name, per_dict in dictionaries:
            dicts.append(map(lambda a: (table_name, a), getattr(self, table_name).split()))

        for split_dicts in zip(*dicts):
            new_me = copy(self)
            for table_name, istorage in split_dicts:
                setattr(new_me, table_name, istorage)
            yield new_me
