import re
import uuid
from collections import namedtuple

import numpy as np

from IStorage import IStorage, AlreadyPersistentError
from hdict import StorageDict
from hecuba import config, log
from hnumpy import StorageNumpy


class StorageObj(object, IStorage):
    args_names = ["name", "tokens", "storage_id", "istorage_props", "class_name"]
    args = namedtuple('StorageObjArgs', args_names)
    _prepared_store_meta = config.session.prepare('INSERT INTO hecuba' +
                                                  '.istorage (storage_id, class_name, name, tokens,istorage_props) '
                                                  ' VALUES (?,?,?,?,?)')
    """
    This class is where information will be stored in Hecuba.
    The information can be in memory, stored in a python dictionary or local variables, or saved in a
    DB(Cassandra), depending on if it's persistent or not.
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
            class_name, mod_name = IStorage.process_path(class_name)
            mod = __import__(mod_name, globals(), locals(), [class_name], 0)

            so = getattr(mod, class_name)(new_args.name.encode('utf8'), new_args.tokens,
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
                                   [storage_args.storage_id,
                                    storage_args.class_name,
                                    storage_args.name,
                                    storage_args.tokens,
                                    storage_args.istorage_props])
        except Exception as ex:
            log.warn("Error creating the StorageDict metadata: %s, %s", str(storage_args), ex)
            raise ex

    def __init__(self, name=None, tokens=None, storage_id=None, istorage_props=None, **kwargs):
        """
            Creates a new storageobj.
            Args:
                name (string): the name of the Cassandra Keyspace + table where information can be found
                tokens (list of tuples): token ranges assigned to the new StorageObj
                storage_id (string):  an unique storageobj identifier
                istorage_props dict(string,string): a map with the storage id of each contained istorage object.
                kwargs: more optional parameters
        """
        log.debug("CREATED StorageObj(%s)", name)
        self._is_persistent = False
        self._persistent_dicts = []
        self._storage_objs = []

        if name is None:
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

        self._storage_id = storage_id
        self._istorage_props = istorage_props

        self._class_name = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)
        if name is not None:
            self._build_args = self.args(self._ksp + '.' + self._table,
                                         self._tokens,
                                         self._storage_id,
                                         self._istorage_props,
                                         self._class_name)

        dictionaries = filter(lambda (k, t): t['type'] == 'StorageDict', self._persistent_props.iteritems())
        for table_name, per_dict in dictionaries:
            dict_name = "%s.%s" % (self._ksp, table_name)

            if istorage_props is not None and dict_name in istorage_props:
                args = config.session.execute(IStorage._select_istorage_meta, (istorage_props[dict_name],))[0]
                # The internal objects must have the same father's tokens
                args = args._replace(tokens=self._tokens)
                log.debug("CREATING INTERNAL StorageDict with %s", args)
                pd = StorageDict.build_remotely(args)
            else:
                if 'indexed_values' in per_dict:
                    indexed_args = per_dict['indexed_values']
                else:
                    indexed_args = None
                pd = StorageDict(None, per_dict['primary_keys'], per_dict['columns'],
                                 tokens=self._tokens, indexed_args=indexed_args)
            setattr(self, table_name, pd)
            self._persistent_dicts.append(pd)

        storageobjs = filter(lambda (k, t): t['type'] not in IStorage._basic_types and t['type'] != 'StorageDict',
                             self._persistent_props.iteritems())
        for table_name, per_dict in storageobjs:
            so_name = "%s.%s" % (self._ksp, table_name)
            cname, module = IStorage.process_path(per_dict['type'])
            mod = __import__(module, globals(), locals(), [cname], 0)
            so = getattr(mod, cname)()
            setattr(self, table_name, so)

        if name is not None:
            self.make_persistent(name)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.getID() == other.getID()

    _dict_case = re.compile('.*@ClassField +(\w+) +dict+ *< *< *([\w:, ]+)+ *> *, *([\w+:., <>]+) *>')
    _tuple_case = re.compile('.*@ClassField +(\w+) +tuple+ *< *([\w, +]+) *>')
    _index_vars = re.compile('.*@Index_on *([A-z0-9]+) +([A-z0-9, ]+)')

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
                    match = IStorage._data_type.match(key)
                    if match is not None and match.lastindex > 1:
                        # an IStorage with a name
                        name, value = match.groups()
                    elif ':' in key:
                        raise SyntaxError
                    else:
                        name = "key" + str(ind)
                        value = key

                    name = name.replace(' ', '')
                    primary_keys.append((name, IStorage._conversions[value]))
                dict_values = dict_values.replace(' ', '')
                if dict_values.startswith('dict'):
                    n = IStorage._sub_dict_case.match(dict_values[4:])
                    # Matching @ClassField of a sub dict
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
                        primary_keys2.append((name, IStorage._conversions[value]))
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
                        columns2.append((name, IStorage._conversions[value]))
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
                        columns.append((name, IStorage._conversions[value]))
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
                            columns.append((name, IStorage._conversions[value]))
                        except KeyError:
                            columns.append((name, value))
                if table_name in this:
                    this[table_name].update({'type': 'StorageDict', 'primary_keys': primary_keys, 'columns': columns})
                else:
                    this[table_name] = {
                        'type': 'StorageDict',
                        'primary_keys': primary_keys,
                        'columns': columns}
            else:
                m = StorageObj._tuple_case.match(line)
                if m is not None:
                    table_name, simple_type = m.groups()
                    simple_type = simple_type.replace(' ', '')
                    simple_type_split = simple_type.split(',')
                    conversion = ''
                    for ind, val in enumerate(simple_type_split):
                        if ind == 0:
                            conversion += IStorage._conversions[val]
                        else:
                            conversion += "," + IStorage._conversions[val]
                    this[table_name] = {
                        'type': 'tuple',
                        'columns': conversion
                    }
                else:
                    m = IStorage._list_case.match(line)
                    if m is not None:
                        table_name, simple_type = m.groups()

                        try:
                            conversion = IStorage._conversions[simple_type]
                        except KeyError:
                            conversion = simple_type
                        this[table_name] = {
                            'type': 'list',
                            'columns': conversion
                        }
                    else:
                        m = IStorage._val_case.match(line)
                        if m is not None:
                            table_name, simple_type = m.groups()
                            this[table_name] = {
                                'type': IStorage._conversions[simple_type]
                            }
                        else:
                            m = IStorage._so_val_case.match(line)
                            if m is not None:
                                table_name, simple_type = m.groups()
                                if simple_type == 'numpy.ndarray':
                                    simple_type = 'hecuba.hnumpy.StorageNumpy'
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
                name (string): name with which the table in the DB will be created
        """
        if self._is_persistent:
            raise AlreadyPersistentError("This StorageObj is already persistent [Before:{}.{}][After:{}]",
                                         self._ksp, self._table, name)
        self._is_persistent = True
        (self._ksp, self._table) = self._extract_ks_tab(name)
        if self._storage_id is None:
            self._storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, self._ksp + '.' + self._table)

        self._build_args = self.args(self._ksp + '.' + self._table,
                                     self._tokens,
                                     self._storage_id,
                                     self._istorage_props,
                                     self._class_name)

        log.info("PERSISTING DATA INTO %s %s", self._ksp, self._table)

        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (self._ksp, config.replication)
        config.session.execute(query_keyspace)

        query_simple = 'CREATE TABLE IF NOT EXISTS ' + str(self._ksp) + '.' + str(self._table) + \
                       '( storage_id uuid PRIMARY KEY, '
        for key, entry in self._persistent_props.items():
            query_simple += str(key) + ' '
            if entry['type'] != 'dict' and entry['type'] in IStorage._valid_types:
                if entry['type'] == 'list' or entry['type'] == 'tuple':
                    query_simple += entry['type'] + '<' + entry['columns'] + '>, '
                else:
                    query_simple += entry['type'] + ', '
            else:
                query_simple += 'uuid, '
        config.session.execute(query_simple[:-2] + ' )')

        for obj_name, obj_info in self._persistent_props.items():
            if hasattr(self, obj_name):
                pd = getattr(self, obj_name)
                if obj_info['type'] not in IStorage._basic_types:
                    sd_name = self._ksp + "." + self._table + "_" + obj_name
                    pd.make_persistent(sd_name)
                setattr(self, obj_name, pd)  # super(StorageObj, self).__setattr__(obj_name, pd) why?

        self._store_meta(self._build_args)

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

        if hasattr(self, '_persistent_dicts'):
            for pers_dict in self._persistent_dicts:
                pers_dict.delete_persistent()

        if hasattr(self, '_storage_objs'):
            for so in self._storage_objs:
                so.delete_persistent()

        query = "TRUNCATE TABLE %s.%s;" % (self._ksp, self._table)
        log.debug("DELETE PERSISTENT: %s", query)
        config.session.execute(query)

    def __getattr__(self, attribute):
        """
            Given an attribute, this function returns the value, obtaining it from either:
            a) memory
            b) the Database
            Args:
                attribute: name of the value that we want to obtain
            Returns:
                value: obtained value
        """
        if attribute[0] != '_' and self._is_persistent and attribute in self._persistent_attrs:
            try:
                query = "SELECT %s FROM %s.%s WHERE storage_id = %s;" \
                        % (attribute, self._ksp,
                           self._table,
                           self._storage_id)
                log.debug("GETATTR: %s", query)
                result = config.session.execute(query)
                for row in result:
                    for row_key, row_var in vars(row).iteritems():
                        if row_var is not None:
                            if isinstance(row_var, list) and isinstance(row_var[0], unicode):
                                new_toreturn = []
                                for entry in row_var:
                                    new_toreturn.append(str(entry))
                                return new_toreturn
                            else:
                                return row_var
                        else:
                            raise AttributeError
            except Exception as ex:
                log.warn("GETATTR ex %s", ex)
                raise AttributeError('value not found')
        else:
            return object.__getattribute__(self, attribute)

    _tablename_finder = re.compile('([A-z0-9_]+)_([0-9]+)$')

    def __setattr__(self, attribute, value):
        """
            Given a key and its value, this function saves it (depending on if it's persistent or not):
                a) In memory
                b) In the DB
            Args:
                attribute: name of the value that we want to set
                value: value that we want to save
        """
        if attribute[0] is '_':
            object.__setattr__(self, attribute, value)
            return

        if attribute in self._persistent_attrs:
            if config.hecuba_type_checking and value is not None and not isinstance(value, dict) and \
                            IStorage._conversions[value.__class__.__name__] != self._persistent_props[attribute][
                        'type']:
                raise TypeError

            if not isinstance(value, IStorage):
                if isinstance(value, np.ndarray):
                    value = StorageNumpy(value)
                elif isinstance(value, dict):
                    per_dict = self._persistent_props[attribute]
                    indexed_args = per_dict.get('indexed_values', None)
                    new_value = StorageDict(None, per_dict['primary_keys'], per_dict['columns'],
                                            tokens=self._tokens, indexed_args=indexed_args)
                    new_value.update(value)
                    value = new_value

            if self._is_persistent:
                if issubclass(value.__class__, IStorage):
                    if not value._is_persistent:
                        value.make_persistent(self._ksp + '.' + self._table + '_' + attribute)
                    values = [self._storage_id, value._storage_id]
                else:
                    values = [self._storage_id, value]

                query = "INSERT INTO %s.%s (storage_id,%s)" % (self._ksp, self._table, attribute)
                query += " VALUES (%s,%s)"

                log.debug("SETATTR: ", query)
                config.session.execute(query, values)

        object.__setattr__(self, attribute, value)

    def __delattr__(self, item):
        """
        Method that deletes a given attribute from a StorageObj
        Args:
            item: the name of the attribute to be deleted
        """
        if self._is_persistent and item in self._persistent_attrs:
            query = "UPDATE %s.%s SET %s = null WHERE storage_id = %s" \
                    % (self._ksp, self._table, item, self._storage_id)
            config.session.execute(query)

        object.__delattr__(self, item)
