import re
import uuid

from IStorage import IStorage
from hecuba import config
from hdict import StorageDict


class StorageObj(object, IStorage):
    """
    This class is where information will be stored in Hecuba.
    The information can be in memory, stored in a python dictionary or local variables, or saved in a
    DDBB(Cassandra), depending on if it's persistent or not.
    """

    @staticmethod
    def build_remotely(results):
        """
        Launches the StorageObj.__init__ from the api.getByID
        Args:
            results: a list of all information needed to create again the storageobj
        Returns:
            so: the created storageobj
        """
        class_name = results.class_name
        if class_name is 'StorageObj':
            so = StorageObj(results.ksp + "." + results.tab, myuuid=results.blockid.encode('utf8'))

        else:
            last = 0
            for key, i in enumerate(class_name):
                if i == '.' and key > last:
                    last = key
            module = class_name[:last]
            cname = class_name[last + 1:]
            mod = __import__(module, globals(), locals(), [cname], 0)
            so = getattr(mod, cname)(results.ksp.encode('utf8') + "." + results.tab.encode('utf8'),
                                     myuuid=results.object_id.encode('utf8'))

        so._objid = results.object_id.encode('utf8')
        return so

    def __init__(self, name=None, myuuid=None):
        """
        Creates a new storageobj.
        Args:
            name (string): the name of the Cassandra Keyspace + table where information can be found
            myuuid (string):  an unique storageobj identifier
        """
        self._persistent_dicts = []
        self._attr_to_column = {}

        self._needContext = True
        self._persistent_props = self._parse_comments(self.__doc__)

        if myuuid is None:
            self._myuuid = str(uuid.uuid1())
        else:
            self._myuuid = myuuid

        if name is None:
            self._persistent = False
        else:
            self._persistent = True
            sp = name.split(".")
            if len(sp) == 2:
                self._ksp = sp[0]
                self._table = sp[1]
                props = list(self._persistent_props)
                self._persistent_props[sp[1]] = self._persistent_props.pop(props[0])
            else:
                self._ksp = config.execution_name
                self._table = name
            class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)

            config.session.execute('INSERT INTO hecuba.storage_objs (object_id, class_name, ksp, tab, obj_type)' +
                                   ' VALUES (%s,%s,%s,%s,%s)',
                                   [self._myuuid, class_name, self._ksp, self._table, 'hecuba'])
            self._get_by_name()
            if myuuid is None:
                query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = {'class': 'SimpleStrategy'," \
                                 "'replication_factor': %d }" % (self._ksp, config.repl_factor)
                try:
                    config.session.execute(query_keyspace)
                except Exception:
                    print "Error executing query:", query_keyspace
                for dict_name, props in self._persistent_props.iteritems():
                    our_values = props['columns']
                    if 'type' in our_values and our_values['type'] == 'dict':
                        keys = map(lambda a: a[0] + ' ' + a[1], props['primary_keys'])
                        columns1 = map(lambda a: a[0] + ' ' + a[1], our_values['primary_keys'])
                        columns2 = map(lambda a: a[0] + ' ' + a[1], our_values['columns'])
                        columns = keys + columns1 + columns2
                    else:
                        columns = map(lambda a: '%s %s' % a, props['primary_keys'] + props['columns'])
                    pks = map(lambda a: a[0], props['primary_keys'])
                    query_table = "CREATE TABLE IF NOT EXISTS %s.%s (%s, PRIMARY KEY (%s));" % (self._ksp, self._table,
                                                                                                str.join(',', columns),
                                                                                                str.join(',', pks))
                    try:
                        config.session.execute(query_table)
                    except config.cluster.NoHostAvailable:
                        print "Error executing query:", query_table

                create_attrs = "CREATE TABLE IF NOT EXISTS %s.%s_attribs (" % (self._ksp, self._table) + \
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

                class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)

                config.session.execute('INSERT INTO hecuba.storage_objs (object_id, class_name, ksp, tab, obj_type)' +
                                       ' VALUES (%s,%s,%s,%s,%s)',
                                       [self._myuuid, class_name, self._ksp, self._table, 'hecuba'])

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
    _conversions = {'atomicint': 'counter',
                    'str': 'text',
                    'bool': 'boolean',
                    'decimal': 'decimal',
                    'float': 'double',
                    'int': 'int',
                    'tuple': 'list',
                    'list': 'list',
                    'generator': 'list',
                    'frozenset': 'set',
                    'set': 'set',
                    'dict': 'map',
                    'long': 'bigint',
                    'buffer': 'blob',
                    'bytearray': 'blob',
                    'counter': 'counter'}

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

    def _get_by_name(self):
        """
        When running the StorageObj.__init__ function with parameters, it retrieves the persistent dict from the DDBB,
        by creating a PersistentDict which links to the Cassandra columnfamily
        """

        props = self._persistent_props
        dictionaries = filter(lambda (k, t): t['type'] == 'dict', props.iteritems())
        for table_name, per_dict in dictionaries:
            pk_names = map(lambda a: a[0], per_dict['primary_keys'])
            col_names = map(lambda a: a[0], per_dict['columns'])
            pd = StorageDict(self._ksp, self._table, pk_names, col_names, is_persistent=self._persistent)
            setattr(self, table_name, pd)
            self._persistent_dicts.append(pd)

    def make_persistent(self, name):
        """
        Once a StorageObj has been created, it can be made persistent. This function retrieves the information about
        the Object class schema, and creates a Cassandra table with those parameters, where information will be saved
        from now on, until execution finishes or StorageObj is no longer persistent.
        It also inserts into the new table all information that was in memory assigned to the StorageObj prior to this
        call.
        """
        sp = name.split(".")
        if len(sp) == 2:
            self._ksp = sp[0]
            self._table = sp[1]
        else:
            self._ksp = config.execution_name
            self._table = name

        props = list(self._persistent_props)
        self._persistent_props[self._table] = self._persistent_props.pop(props[0])

        self._get_by_name()

        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = {'class': 'SimpleStrategy'," \
                         "'replication_factor': %d }" % (self._ksp, config.repl_factor)
        try:
            config.session.execute(query_keyspace)
        except Exception:
            print "Error executing query:", query_keyspace
        for dict_name, props in self._persistent_props.iteritems():
            our_values = props['columns']
            if 'type' in our_values and our_values['type'] == 'dict':
                keys = map(lambda a: a[0] + ' ' + a[1], props['primary_keys'])
                columns1 = map(lambda a: a[0] + ' ' + a[1], our_values['primary_keys'])
                columns2 = map(lambda a: a[0] + ' ' + a[1], our_values['columns'])
                columns = keys + columns1 + columns2
            else:
                columns = map(lambda a: '%s %s' % a, props['primary_keys'] + props['columns'])
            pks = map(lambda a: a[0], props['primary_keys'])
            query_table = "CREATE TABLE IF NOT EXISTS %s.%s (%s, PRIMARY KEY (%s));" % (self._ksp, self._table,
                                                                                        str.join(',', columns),
                                                                                        str.join(',', pks))
            try:
                config.session.execute(query_table)
            except config.cluster.NoHostAvailable:
                print "Error executing query:", query_table

        empty_table = "TRUNCATE " + str(self._ksp) + "." + str(self._table) + ";"
        config.session.execute(empty_table)

        classname = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)

        config.session.execute('INSERT INTO hecuba.storage_objs (object_id, class_name, ksp, tab, obj_type)' +
                               ' VALUES (%s,%s,%s,%s,%s)',
                               [self._myuuid, classname, self._ksp, self._table, 'hecuba'])

        create_attrs = "CREATE TABLE IF NOT EXISTS %s.%s_attribs (" % (self._ksp, self._table) + \
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

        empty_attrs = "TRUNCATE " + str(self._ksp) + "." + str(self._table) + "_attribs;"
        config.session.execute(empty_attrs)

        self._persistent = True
        for dict in self._persistent_dicts:
            memory_vals = dict.iteritems()
            dict.is_persistent = True
            for key, val in memory_vals:
                dict[key] = val

        for key, variable in vars(self).iteritems():
            if not key[0] == '_':
                insert_query = 'INSERT INTO ' + str(self._ksp) + '.' + str(self._table) + '_attribs (name, '
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

    def increment(self, target, value):
        """
        Instead of increasing the existing value in position target, it sets it to the desired value (only intended to
        be used with counter tables
        """
        self[target] = value

    def stop_persistent(self):
        """
            Empties the Cassandra table where the persistent StorageObj stores data
        """
        def_dict = self._get_default_dict()
        def_dict.dictCache.cache = {}

        query = "TRUNCATE %s.%s;" % (self._ksp, self._table)
        config.session.execute(query)
        for d in self._persistent_dicts:
            query = "TRUNCATE %s.%s;" % (self._ksp, d._table)
            config.session.execute(query)

    def delete_persistent(self):
        """
            Deletes the Cassandra table where the persistent StorageObj stores data
        """
        def_dict = self._get_default_dict()
        def_dict.dictCache.cache = {}

        self._persistent = False
        for dics in self._persistent_dicts:
            dics.is_persistent = False

        query = "DROP COLUMNFAMILY %s.%s;" % (self._ksp, self._table)
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

        if key[0] != '_' and self._persistent:
            try:
                result = config.session.execute(
                    "SELECT * FROM " + str(self._ksp) + "." + str(self._table) + "_attribs WHERE name = \'" + str(
                        key) + "\';")
                for row in result:
                    for rowkey, rowvar in vars(row).iteritems():
                        if not rowkey == 'name':
                            if rowvar is not None:
                                if rowkey == 'doubletuple' or rowkey == 'inttuple' or rowkey == 'texttuple':
                                    return tuple(rowvar)
                                else:
                                    return rowvar
            except:
                col_name = self._attr_to_column[key]
                query = "SELECT " + col_name + " FROM " + self._ksp + "." + self._table + " WHERE name = %s"
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

        if key[0] is not '_' and self._persistent:
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

            querytable = "INSERT INTO " + self._ksp + "." + self._table + "_attribs (name," + self._attr_to_column[
                key] + \
                         ") VALUES (%s,%s)"

            config.session.execute(querytable, [key, value])

        else:
            super(StorageObj, self).__setattr__(key, value)

    def getID(self):
        """
        This function returns the ID of the StorageObj
        """
        return '%s_1' % self._myuuid

    def split(self):
        """
          Depending on if it's persistent or not, this function returns the list of keys of the PersistentDict
          assigned to the StorageObj, or the list of keys
          Returns:
               a) List of keys in case that the SO is not persistent
               b) Iterator that will return Blocks, one by one, where we can find the SO data in case it's persistent
        """

        return self
