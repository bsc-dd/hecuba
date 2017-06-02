import re
from collections import namedtuple
from copy import copy
import uuid
from IStorage import IStorage
from hdict import StorageDict
from hecuba import config, log
import numpy as np


class StorageObj(object, IStorage):
    args_names = ["name", "tokens", "storage_id", "istorage_props", "class_name"]
    args = namedtuple('StorageObjArgs', args_names)
    _prepared_store_meta = config.session.prepare('INSERT INTO ' + config.execution_name + '.istorage '
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
        """
            Saves the information of the object in the istorage table.
            Args:
                storage_args (object): contains all data needed to restore the object from the workers
        """
        log.debug("StorageObj: storing media %s", storage_args)
        try:
            config.session.execute(StorageObj._prepared_store_meta,
                                   [storage_args.storage_id, storage_args.class_name,
                                    storage_args.name + '_' + str(storage_args.storage_id).replace('-', ''),
                                    storage_args.tokens, storage_args.istorage_props])
        except Exception as ex:
            print "Error creating the StorageDict metadata:", storage_args, ex
            raise ex

    def __init__(self, name=None, tokens=None, storage_id=None, istorage_props=None, **kwargs):
        """
            Creates a new storageobj.
            Args:
                name (string): the name of the Cassandra Keyspace + table where information can be found
                tokens (list of tuples): token ranges assigned to the new StorageObj
                storage_id (string):  an unique storageobj identifier
                istorage_props dict(string,string) a map with the storage id of each contained istorage object.
        """
        log.debug("CREATED StorageObj(%s)", name)
        self._is_persistent = False
        self._persistent_dicts = []
        self._attr_to_column = {}
        if name is None:
            self._name = name
            self._ksp = config.execution_name
        else:
            (self._ksp, self._table) = self._extract_ks_tab(name)

        self._persistent_props = StorageObj._parse_comments(self.__doc__)
        self._persistent_attrs = self._persistent_props.keys()

        if tokens is None:
            # log.info('using all tokens')
            tokens = map(lambda a: a.value, config.cluster.metadata.token_map.ring)
            self._tokens = IStorage._discrete_token_ranges(tokens)
        else:
            self._tokens = tokens

        if storage_id is not None:
            self._storage_id = storage_id
        elif name is not None:
            self._storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, self._ksp + '.' + self._table)
        elif storage_id is None:
            self._storage_id = None

        self._istorage_props = istorage_props

        self._class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)
        if name is not None:
            self._build_args = self.args(self._ksp + '.' + self._table,
                                         self._tokens,
                                         self._storage_id,
                                         self._istorage_props,
                                         self._class_name)

        dictionaries = filter(lambda (k, t): t['type'] == 'dict', self._persistent_props.iteritems())
        for table_name, per_dict in dictionaries:
            dict_name = "%s.%s" % (self._ksp, table_name)

            dict_case = True
            if istorage_props is not None and dict_name in istorage_props:
                args = config.session.execute(IStorage._select_istorage_meta, (istorage_props[dict_name],))[0]
                # The internal objects must have the same father's tokens
                args = args._replace(tokens=self._tokens)
                log.debug("CREATING INTERNAL StorageDict with %s", args)
                pd = StorageDict.build_remotely(args)
            else:
                for ind, entry in enumerate(per_dict['columns']):
                    if entry[1] not in IStorage.valid_types and '<' not in entry[1]:
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
                            pd = getattr(mod, c_name)(name=self._ksp + '.' + so_name)
                        else:
                            mod = __import__(self.__module__, globals(), locals(), [field_type], 0)
                            pd = getattr(mod, field_type)(name=self._ksp + '.' + so_name)
                        dict_case = False
            pd = StorageDict(per_dict['primary_keys'], per_dict['columns'], tokens=self._tokens)
            pd._primary_keys = per_dict['primary_keys']
            pd._columns = per_dict['columns']
            pd._tokens = self._tokens
            setattr(self, table_name, pd)

            if dict_case:
                self._persistent_dicts.append(pd)

        sos = filter(lambda (k, t): t['type'] not in IStorage.valid_types and '<' not in t['type'],
                     self._persistent_props.iteritems())
        for so_name, so in sos:
            if not hasattr(self, so_name):
                field_type = so['type']
                if '.' in field_type and not field_type == 'numpy.ndarray':
                    last = 0
                    for key, i in enumerate(field_type):
                        if i == '.' and key > last:
                            last = key
                    module = field_type[:last]
                    c_name = field_type[last + 1:]
                    mod = __import__(module, globals(), locals(), [c_name], 0)
                    my_so = getattr(mod, c_name)(self._ksp + '.' + so_name)
                else:
                    mod = __import__(self.__module__, globals(), locals(), [field_type], 0)
                    my_so = getattr(mod, field_type)(self._ksp + '.' + so_name)
                setattr(self, so_name, my_so)

        if name is not None:
            self.make_persistent(name)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.getID() == other.getID()

    _valid_type = '(atomicint|str|bool|decimal|float|int|tuple|list|generator|frozenset|set|dict|long|buffer|' \
                  'numpy.ndarray|counter)'
    _data_type = re.compile('(\w+) *: *%s' % _valid_type)
    _so_data_type = re.compile('(\w+)*:(\w.+)')
    _dict_case = re.compile('.*@ClassField +(\w+) +dict+ *< *< *([\w:, ]+)+ *> *, *([\w+:., <>]+) *>')
    _tuple_case = re.compile('.*@ClassField +(\w+) +tuple+ *< *([\w, +]+) *>')
    _list_case = re.compile('.*@ClassField +(\w+) +list+ *< *([\w:\.+]+) *>')
    _sub_dict_case = re.compile(' *< *< *([\w:, ]+)+ *> *, *([\w+:, <>]+) *>')
    _sub_tuple_case = re.compile(' *< *([\w:, ]+)+ *>')
    _val_case = re.compile('.*@ClassField +(\w+) +%s' % _valid_type)
    _so_val_case = re.compile('.*@ClassField +(\w+) +([\w.]+)')
    _index_vars = re.compile('.*@Index_on+\s*([A-z,]+)+([A-z, ]+)')

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
            m = StorageObj._dict_case.match(line)
            if m is not None:
                # Matching @ClassField of a dict
                table_name, dict_keys, dict_values = m.groups()
                primary_keys = []
                for ind, key in enumerate(dict_keys.split(",")):
                    try:
                        name, value = StorageObj._data_type.match(key).groups()
                    except re.error:
                        if ':' in key:
                            raise SyntaxError
                        else:
                            name = "key" + str(ind)
                            value = key
                    name = name.replace(' ', '')
                    primary_keys.append((name, StorageObj._conversions[value]))
                dict_values = dict_values.replace(' ', '')
                if dict_values.startswith('dict'):
                    n = StorageObj._sub_dict_case.match(dict_values[4:])
                    # Matching @ClassField of a sub dict
                    dict_keys2, dict_values2 = n.groups()
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
                    dict_values2 = dict_values2.replace(' ', '')
                    if dict_values2.startswith('tuple'):
                        dict_values2 = dict_values2[6:]
                    for ind, val in enumerate(dict_values2.split(",")):
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
                elif dict_values.startswith('tuple'):
                    n = StorageObj._sub_tuple_case.match(dict_values[5:])
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
                        columns.append((name, StorageObj._conversions[value]))
                else:
                    columns = []
                    for ind, val in enumerate(dict_values.split(",")):
                        try:
                            name, value = StorageObj._data_type.match(val).groups()
                        except re.error:
                            if ':' in val:
                                name, value = StorageObj._so_data_type.match(val).groups()
                            else:
                                name = "val" + str(ind)
                                value = val
                        name = name.replace(' ', '')
                        try:
                            columns.append((name, StorageObj._conversions[value]))
                        except KeyError:
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
                    simple_type = simple_type.replace(' ','')
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

                        try:
                            conversion = StorageObj._conversions[simple_type]
                        except KeyError:
                            conversion = simple_type
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
            the Object class schema, and creates a Cassandra table with those parameters, where information will be
            saved from now on, until execution finishes or StorageObj is no longer persistent.
            It also inserts into the new table all information that was in memory assigned to the StorageObj prior to
            this call.
            Args:
                name (string): name with which the table in the DDBB will be created
        """
        self._is_persistent = True
        (self._ksp, self._table) = self._extract_ks_tab(name)
        self._storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, self._ksp + '.' + self._table)

        self._build_args = self.args(self._ksp + '.' + self._table,
                                     self._tokens,
                                     self._storage_id,
                                     self._istorage_props,
                                     self._class_name)

        log.info("PERSISTING DATA INTO %s %s", self._ksp, self._table)

        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = {'class': 'SimpleStrategy'," \
                         "'replication_factor': %d }" % (self._ksp, config.repl_factor)
        try:
            config.session.execute(query_keyspace)
        except Exception as ex:
            print "Error executing query:", query_keyspace
            raise ex
        valid_types = ['counter', 'text', 'boolean', 'decimal', 'double', 'int', 'tuple', 'list', 'set', 'map',
                       'bigint', 'blob', 'map']
        query_simple = 'CREATE TABLE IF NOT EXISTS ' + \
                       str(self._ksp) + '.' + str(self._table) + '_' + str(self._storage_id).replace('-', '') + \
                       '( storage_id uuid PRIMARY KEY, '
        for key, entry in self._persistent_props.iteritems():
            query_simple += str(key) + ' '
            if '<' in entry:
                in_entry = entry['type'].split('<')[1].replace('>', '')
            else:
                in_entry = 'int'
            if entry['type'] == 'dict' or \
                            entry['type'] == 'tuple' or \
                            in_entry not in valid_types or \
                            entry['type'].split('<')[0] not in valid_types:
                query_simple += 'uuid, '
            else:
                query_simple += entry['type'] + ', '
                if entry['type'] == 'blob':
                    query_simple += str(key) + "_size text, "
                    query_simple += str(key) + "_shape text, "
        config.session.execute(query_simple[0:len(query_simple) - 2] + ' )')

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
        '''
        others = filter(lambda (k, t): t['type'] not in valid_types, self._persistent_props.iteritems())
        is_props = self._build_args.istorage_props
        if is_props is None:
            is_props = {}
        changed = False
        for table_name, _ in others:
            changed = True
            pd = getattr(self, table_name)
            name2 = self._ksp + "." + table_name
            pd.make_persistent(name2)
            is_props[name2] = str(pd._storage_id)
        '''

        if changed or self._storage_id is None:
            if self._storage_id is None:
                self._storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, name)
                self._build_args = self._build_args._replace(storage_id=self._storage_id, istorage_props=is_props)
        self._store_meta(self._build_args)

        names = "storage_id"
        values = list()
        values.append(self._storage_id)
        for key, variable in vars(self).iteritems():
            to_insert = False
            if not key[0] == '_':
                if key in self._persistent_attrs:
                    to_insert = True
                    if issubclass(variable.__class__, IStorage):
                        names += ", " + str(key)
                        if variable._storage_id is None:
                            variable._storage_id = uuid.uuid3(uuid.NAMESPACE_DNS,
                                                              variable._ksp + '.' + variable._table)
                        values.append(variable._storage_id)
                    else:
                        names += ", " + str(key)
                        if type(variable) is str:
                            values.append(str(variable))
                        elif type(variable) is np.ndarray:
                            values.append(variable.tostring())
                            query2 = "INSERT INTO %s.%s (storage_id,%s_size,%s_shape) VALUES (?,?,?)" \
                                     % (self._ksp,
                                        str(self._table).lower() + '_' + str(self._storage_id).replace('-', ''),
                                        key, key)
                            prepared2 = config.session.prepare(query2)
                            config.session.execute(prepared2,
                                                   [self._storage_id, str(variable.dtype), str(variable.shape)])
                        else:
                            values.append(variable)
            if to_insert:
                query = "INSERT INTO " + \
                        str(self._ksp) + '.' + str(self._table) + \
                        '_' + str(self._storage_id).replace('-', '') + " (" + names + ") VALUES (?, ?)"
                prepared = config.session.prepare(query)
                config.session.execute(prepared,
                                       values)

    def stop_persistent(self):
        """
            The StorageObj stops being persistent, but keeps the information already stored in Cassandra
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
        """
            Returns the value of the attribute with name 'key'
            Args:
                key (string): name of the attribute from which we want to obtain the value
            Returns:
                to_return (not defined): the value stored in the attribute 'key'.
        """
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
            if key in self._persistent_attrs:
                try:
                    query = "SELECT " + str(key) + " FROM %s.%s WHERE storage_id = %s;" \
                                                   % (self._ksp,
                                                      str(self._table).lower() + '_' + str(self._storage_id).replace(
                                                          '-', ''),
                                                      self._storage_id)
                    log.debug("GETATTR: %s", query)
                    result = config.session.execute(query)
                    for row in result:
                        for row_key, row_var in vars(row).iteritems():
                            # if not row_key == 'name':
                            if row_var is not None:
                                to_return = row_var
                                if self._persistent_props[key]['type'] == 'blob':
                                    query2 = "SELECT " + str(key) + "_size, " + str(key) + "_shape FROM %s.%s" \
                                                                                           " WHERE storage_id = %s;" \
                                                                                           % (self._ksp,
                                                                                              str(
                                                                                                  self._table).lower() + '_' + str(
                                                                                                  self._storage_id).replace(
                                                                                                  '-', ''),
                                                                                              self._storage_id)
                                    result2 = config.session.execute(query2)
                                    array_type = ''
                                    array_shape = ''
                                    for row2 in result2:
                                        array_type = getattr(row2, key + "_size")
                                        array_shape = getattr(row2, key + "_shape")
                                    if array_type is not None:
                                        try:
                                            if str(array_shape).split(", ")[1] != ')':
                                                to_return = np.fromstring(to_return, dtype=np.dtype(str(array_type))). \
                                                    reshape(tuple(map(int, array_shape[1:-1].split(','))))
                                            else:
                                                to_return = np.fromstring(to_return, dtype=np.dtype(str(array_type)))
                                        except Exception as e:
                                            print "error:", e
                                return to_return
                except Exception as ex:
                    print "error:", ex
                    log.warn("GETATTR ex %s", ex)
                    raise KeyError('value not found')
            else:
                return object.__getattribute__(self, key)
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
        elif hasattr(self, '_is_persistent') and self._is_persistent:
            if key in self._persistent_attrs:
                query = "INSERT INTO %s.%s (storage_id,%s) VALUES (?,?)" \
                        % (self._ksp, str(self._table).lower() + '_' + str(self._storage_id).replace('-', ''), key)
                prepared = config.session.prepare(query)
                if not type(value) == dict and not type(value) == StorageDict:
                    if type(value) == str:
                        values = [self._storage_id, "" + str(value) + ""]
                    else:
                        if type(value) == np.ndarray:
                            values = [self._storage_id, value.tostring()]
                            query2 = "INSERT INTO %s.%s (storage_id,%s_size,%s_shape) VALUES (?,?,?)" \
                                     % (self._ksp,
                                        str(self._table).lower() + '_' + str(self._storage_id).replace('-', ''),
                                        key, key)
                            prepared2 = config.session.prepare(query2)
                            config.session.execute(prepared2, [self._storage_id, str(value.dtype), str(value.shape)])
                        else:
                            values = [self._storage_id, value]
                else:
                    values = [self._storage_id, str(value._storage_id)]
                log.debug("SETATTR: ", query)
                try:
                    config.session.execute(prepared, values)
                except Exception as e:
                    print "Error setting attribute:", e
            else:
                super(StorageObj, self).__setattr__(key, value)
        else:
            super(StorageObj, self).__setattr__(key, value)

    def getID(self):
        """
            This function returns the ID of the StorageObj
            Returns:
                storage_id of the object, followed by '_1'
        """
        print "self_storage_id getID of SO:", self._storage_id
        return '%s_1' % str(self._storage_id)

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

