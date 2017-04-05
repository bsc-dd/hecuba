import re
from collections import namedtuple
from copy import copy
import uuid

from IStorage import IStorage
from hdict import StorageDict
from hecuba import config, log


class StorageObj(object, IStorage):
    args_names = ["name", "tokens", "storage_id", "istorage_props", "class_name"]
    args = namedtuple('StorageObjArgs', args_names)
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba2.istorage '
                                                  '(storage_id, class_name, name, tokens,istorage_props) '
                                                  ' VALUES (?,?,?,?,?)')
    """
    This class is where information will be stored in Hecuba.
    The information can be in memory, stored in a python dictionary or local variables, or saved in a
    DDBB(Cassandra), depending on if it's persistent or not.
    """

    @staticmethod
    def build_remotely(new_args):
        """
        Launches the StorageObj.__init__ from the uuid api.getByID
        Args:
            new_args: a list of all information needed to create again the storageobj
        Returns:
            so: the created storageobj
        """
        log.debug("Building Storage object with %s", new_args)
        class_name = new_args.class_name
        if class_name is 'StorageObj':
            so = StorageObj(new_args.name.encode('utf8'), new_args.tokens, new_args.storage_id, new_args.istorage_props)

        else:
            last = 0
            for key, i in enumerate(class_name):
                if i == '.' and key > last:
                    last = key
            module = class_name[:last]
            class_name = class_name[last + 1:]
            mod = __import__(module, globals(), locals(), [class_name], 0)
            if str(new_args.name.encode('utf8')).count('_') == 0:
                so = getattr(mod, class_name)(new_args.name.encode('utf8'), new_args.tokens,
                                              new_args.storage_id, new_args.istorage_props)
            else:
                so = getattr(mod, class_name)(str(new_args.name.encode('utf8')).split('_')[0], new_args.tokens,
                                              new_args.storage_id, new_args.istorage_props)

        return so

    @staticmethod
    def _store_meta(storage_args):
        log.debug("StorageObj: storing media %s", storage_args)
        try:
            config.session.execute(StorageObj._prepared_store_meta,
                                   [storage_args.storage_id, storage_args.class_name,
                                    storage_args.name + '_' + str(storage_args.storage_id).replace('-', ''),
                                    storage_args.tokens, storage_args.istorage_props])
        except Exception as ex:
            print "Error creating the StorageDict metadata:", storage_args, ex
            raise ex

    def __init__(self, name=None, tokens=None, storage_id=None, istorage_props=None):
        """
        Creates a new storageobj.
        Args:
            name (string): the name of the Cassandra Keyspace + table where information can be found
            storage_id (string):  an unique storageobj identifier
            istorage_props dict(string,string) a map with the storage id of each contained istorage object.
        """
        log.debug("CREATED StorageObj(%s)", name)
        self._is_persistent = False
        self._persistent_dicts = []
        self._attr_to_column = {}
        if name is None:
            self._name = name
            ksp = config.execution_name
        else:
            self._name = None
            (ksp, _) = self._extract_ks_tab(name)

        self._persistent_props = self._parse_comments(self.__doc__)

        if tokens is None:
            # log.info('using all tokens')
            tokens = map(lambda a: a.value, config.cluster.metadata.token_map.ring)
            self._tokens = IStorage._discrete_token_ranges(tokens)
        else:
            self._tokens = tokens

        if storage_id is not None:
            self._storage_id = storage_id
        elif name is not None:
            self._storage_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, name))
        else:
            self._storage_id = None

        class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)
        self._build_args = self.args(name, self._tokens, self._storage_id, istorage_props, class_name)

        valid_types = ['counter', 'text', 'boolean', 'decimal', 'double', 'int', 'list', 'set', 'map',
                       'bigint', 'blob', 'counter', 'dict']

        dictionaries = filter(lambda (k, t): t['type'] == 'dict', self._persistent_props.iteritems())
        for table_name, per_dict in dictionaries:

            dict_name = "%s.%s" % (ksp, table_name)

            dict_case = True
            if istorage_props is not None and dict_name in istorage_props:
                args = config.session.execute(IStorage._select_istorage_meta, (istorage_props[dict_name],))[0]
                # The internal objects must have the same father's tokens
                args = args._replace(tokens=self._tokens)
                log.debug("CREATING INTERNAL StorageDict with %s", args)
                pd = StorageDict.build_remotely(args)
            else:
                for ind, entry in enumerate(per_dict['columns']):
                    if entry[1] not in valid_types and '<' not in entry[1]:
                        so_name = entry[0]
                        field_type = entry[1]
                        if '.' in field_type:
                            last = 0
                            for key, i in enumerate(field_type):
                                if i == '.' and key > last:
                                    last = key
                            module = field_type[:last]
                            c_name = field_type[last + 1:]
                            mod = __import__(module, globals(), locals(), [c_name], 0)
                            pd = getattr(mod, c_name)(ksp + '.' + so_name)
                        else:
                            mod = __import__(self.__module__, globals(), locals(), [field_type], 0)
                            pd = getattr(mod, field_type)(ksp + '.' + so_name)
                        per_dict['columns'][ind] = (so_name, 'uuid')
                        dict_case = False
            pd = StorageDict(per_dict['primary_keys'], per_dict['columns'], tokens=self._tokens)
            setattr(self, table_name, pd)

            if dict_case:
                self._persistent_dicts.append(pd)

        sos = filter(lambda (k, t): t['type'] not in valid_types and '<' not in t['type'],
                     self._persistent_props.iteritems())
        for so_name, so in sos: 
            field_type = so['type']
            if '.' in field_type:
                last = 0
                for key, i in enumerate(field_type):
                    if i == '.' and key > last:
                        last = key
                module = field_type[:last]
                c_name = field_type[last + 1:]
                mod = __import__(module, globals(), locals(), [c_name], 0)
                my_so = getattr(mod, c_name)(so_name)
            else:
                mod = __import__(self.__module__, globals(), locals(), [field_type], 0)
                my_so = getattr(mod, field_type)(so_name)
            setattr(self, so_name, my_so)

        if name is not None:
            self.make_persistent(name)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.getID() == other.getID()

    _valid_type = '(atomicint|str|bool|decimal|float|int|tuple|list|generator|frozenset|set|dict|long|buffer|' \
                  'bytearray|counter)'
    _data_type = re.compile('(\w+) *: *%s' % _valid_type)
    _so_data_type = re.compile('(\w+)*:(\w.+)')
    _dict_case = re.compile('.*@ClassField +(\w+) +dict +< *< *([\w:, ]+)+ *> *, *([\w+:., <>]+) *>')
    _tuple_case = re.compile('.*@ClassField +(\w+) +tuple+< *([\w,+]+) *>')
    _list_case = re.compile('.*@ClassField +(\w+) +list+< *([\w+]+) *>')
    _sub_dict_case = re.compile(' *< *< *([\w:, ]+)+ *> *, *([\w+:, <>]+) *>')
    _sub_tuple_case = re.compile(' *< *([\w:, ]+)+ *>')
    _val_case = re.compile('.*@ClassField +(\w+) +%s' % _valid_type)
    _so_val_case = re.compile('.*@ClassField +(\w+) +([\w.]+)')
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
                    except Exception as e:
                        print "Error:", e
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
                        except Exception as e:
                            print "Error:", e
                            if ':' in val:
                                name, value = StorageObj._so_data_type.match(val).groups()
                            else:
                                name = "val" + str(ind)
                                value = val
                        name = name.replace(' ', '')
                        try:
                            columns.append((name, StorageObj._conversions[value]))
                        except Exception as e:
                            print "Error:", e
                            columns.append((name, value))
                if table_name in this:
                    this[table_name].update({'type': 'dict', 'primary_keys': primary_keys, 'columns': columns})
                else:
                    this[table_name] = {
                        'type': 'dict',
                        'primary_keys': primary_keys,
                        'columns': columns}
            else:
                m = StorageObj._tuple_case.match(line)
                if m is not None:
                    table_name, simple_type = m.groups()
                    simple_type_split = simple_type.split(',')
                    conversion = ''
                    for ind, val in enumerate(simple_type_split):
                        if ind == 0:
                            conversion += StorageObj._conversions[val]
                        else:
                            conversion += "," + StorageObj._conversions[val]
                    this[table_name] = {
                        'type': 'tuple<' + conversion + '>'
                    }
                else:
                    m = StorageObj._list_case.match(line)
                    if m is not None:
                        table_name, simple_type = m.groups()
                        conversion = StorageObj._conversions[simple_type]
                        this[table_name] = {
                            'type': 'list<' + conversion + '>'
                        }
                    else:
                        m = StorageObj._val_case.match(line)
                        if m is not None:
                            table_name, simple_type = m.groups()
                            this[table_name] = {
                                'type': StorageObj._conversions[simple_type]
                            }
                        else:
                            m = StorageObj._so_val_case.match(line)
                            if m is not None:
                                table_name, simple_type = m.groups()
                                this[table_name] = {
                                    'type': simple_type
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
        # log.info("PERSISTING DATA INTO %s %s", self._ksp, self._table)

        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = {'class': 'SimpleStrategy'," \
                         "'replication_factor': %d }" % (self._ksp, config.repl_factor)
        try:
            config.session.execute(query_keyspace)
        except Exception as ex:
            print "Error executing query:", query_keyspace
            raise ex

        query_simple = 'CREATE TABLE IF NOT EXISTS ' + str(name) + '_' + str(self._storage_id).replace('-', '') + \
                       '( storage_id uuid PRIMARY KEY, '
        for key, entry in self._persistent_props.iteritems():
            query_simple += str(key) + ' '
            if not entry['type'] == 'dict':
                query_simple += entry['type'] + ', '
            else:
                query_simple += 'uuid, '
        config.session.execute(query_simple[0:len(query_simple)-2] + ' )')

        dictionaries = filter(lambda (k, t): t['type'] == 'dict', self._persistent_props.iteritems())
        is_props = self._build_args.istorage_props
        if is_props is None:
            is_props = {}
        changed = False
        for table_name, _ in dictionaries:
            changed = True
            pd = getattr(self, table_name)
            name2 = self._ksp + "." + table_name
            pd.make_persistent(name2)
            is_props[name2] = pd._storage_id

        if changed or self._storage_id is None:
            if self._storage_id is None:
                self._storage_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, name))
                self._build_args = self._build_args._replace(storage_id=self._storage_id, istorage_props=is_props)
            self._store_meta(self._build_args)

        self._is_persistent = True

        to_insert = False
        names = "storage_id"
        values = str(self._storage_id)
        for key, variable in vars(self).iteritems():
            if not key[0] == '_':
                to_insert = True
                if not type(variable) == dict and not type(variable) == StorageDict:
                    names += ", " + str(key)
                    values += ", " + str(variable)
                else:
                    names += ", " + str(key)
                    values += ", " + str(variable._storage_id)
        if to_insert:
            insert_query = "INSERT INTO " + str(name) + '_' + str(self._storage_id).replace('-', '') + \
                              " (" + names + ") VALUES (" + values + ")"
            config.session.execute(insert_query)

    def stop_persistent(self):
        """
            Empties the Cassandra table where the persistent StorageObj stores data
        """

        log.debug("STOP PERSISTENT")
        for persistent_dict in self._persistent_dicts:
            persistent_dict.stop_persistent()

    def delete_persistent(self):
        """
            Deletes the Cassandra table where the persistent StorageObj stores data
        """
        self._is_persistent = False
        for pers_dict in self._persistent_dicts:
            pers_dict.delete_persistent()

        query = "TRUNCATE TABLE %s.%s;" % (self._ksp, self._table)
        log.debug("DELETE PERSISTENT: %s", query)
        config.session.execute(query)

    def __getattribute__(self, key):
        to_return = object.__getattribute__(self, key)
        return to_return

    def __getattr__(self, key):
        """
        Given a key, this function reads the configuration table in order to know if the attribute can be found:
        a) In memory
        b) In the Database
        Args:
            key: name of the value that we want to obtain
        Returns:
            value: obtained value
        """
        if key[0] != '_' and key is not 'storage_id' and self._is_persistent:
            try:
                query = "SELECT " + str(key) + " FROM %s.%s WHERE storage_id = %s;"\
                    % (self._ksp, str(self.__class__.__name__).lower() + '_' + str(self._storage_id).replace('-', ''),
                       self._storage_id)
                log.debug("GETATTR: %s", query)
                result = config.session.execute(query)
                for row in result:
                    for row_key, row_var in vars(row).iteritems():
                        if not row_key == 'name':
                            if row_var is not None:
                                return row_var
                            else:
                                raise KeyError('value not found')

            except Exception as ex:
                log.warn("GETATTR ex %s", ex)
                raise KeyError('value not found')
        else:
            return object.__getattribute__(self, key)

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
            query = "INSERT INTO %s.%s (storage_id,%s)" % (self._ksp, str(self.__class__.__name__).lower(), key)
            if not type(value) == dict and not type(value) == StorageDict:
                query += "VALUES (" + str(self._storage_id) + ", " + str(value) + ")"
            else:
                query += "VALUES (" + str(self._storage_id) + ", " + str(value._storage_id) + ")"
            log.debug("SETATTR: ", query)
            config.session.execute(query)
        else:
            super(StorageObj, self).__setattr__(key, value)

    def getID(self):
        """
        This function returns the ID of the StorageObj
        """
        return '%s_1' % self._storage_id

    def old_split(self):
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

