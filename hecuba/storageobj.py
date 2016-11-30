# author: G. Alomar
import re
import time
import uuid

from hecuba import session, config
from hecuba.dict import PersistentDict
from hecuba.iter import KeyIter


class StorageObj(object):
    """
    This class is where information will be stored in Hecuba.
    The information can be in memory, stored in a python dictionary or local variables, or saved in a
    DDBB(Cassandra), depending on if it's persistent or not.
    """
    nextKeys = []
    cntxt = ''

    @staticmethod
    def build_remotely(results):
        """
        Launches the StorageObj.__init__ from the api.getByID
        Args:
            results: a list of all information needed to create again the storageobj
        Returns:
            so: the created storageobj
        """
        classname = results.storageobj_classname
        if classname is 'StorageObj':
            so = StorageObj(results.ksp+"."+results.tab, myuuid=results.blockid)

        else:
            last = 0
            for key, i in enumerate(classname):
                if i == '.' and key > last:
                    last = key
            module = classname[:last]
            cname = classname[last + 1:]
            mod = __import__(module, globals(), locals(), [cname], 0)
            so = getattr(mod, cname)(results.ksp+"."+results.tab, myuuid=results.blockid)

        so._objid = results.blockid
        return so

    def __init__(self, name=None, myuuid=None):
        """
        Creates a new storageobj.

        Args:
            ksp (string): the Cassandra keyspace where information can be found
            table (string): the name of the Cassandra collection/table where information can be found
            myuuid (string):  an unique storageobj identifier
        """
        self._persistent_dicts = []
        self._attr_to_column = {}
        if name is None:
            self._ksp = config.execution_name
            self._table = self.__class__.__name__.lower()
            self._persistent = False
        else:
            self._persistent = True
            sp = name.split(".")
            if len(sp) == 2:
                self._table = sp[1]
                self._ksp = sp[0]
            else:
                self._table = name
                self._ksp = config.execution_name


        if myuuid is None:
            self._myuuid = str(uuid.uuid1())
            classname = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)

            session.execute('INSERT INTO hecuba.blocks (blockid, storageobj_classname, ksp, tab, obj_type)' +
                            ' VALUES (%s,%s,%s,%s,%s)',
                            [self._myuuid, classname, self._ksp, self._table, 'hecuba'])
        else:
            self._myuuid = myuuid
        self._needContext = True
        if not hasattr(self.__class__, '_persistent_props'):
            self.__class__._persistent_props = self._parse_comments(self.__doc__)
        self.getByName()

    _valid_type = '(atomicint|str|bool|decimal|float|int|tuple|list|generator|frozenset|set|dict|long|buffer|bytearray|counte)'
    _data_type = re.compile('(\w+) *: *%s' % _valid_type)
    _dict_case = re.compile('.*@ClassField +(\w+) +dict +< *< *([\w:]+) *> *, *([\w+:]+) *>.*')
    _val_case = re.compile('.*@ClassField +(\w+) +(\w+) +%s' % _valid_type)
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
        this = {}
        for line in comments.split('\n'):
            m = StorageObj._dict_case.match(line)
            if m is not None:
                # Matching @ClassField of a dict
                table_name, dict_keys, dict_vals = m.groups()
                primary_keys = []
                for key in dict_keys.split(","):
                    name, value = StorageObj._data_type.match(key).groups()
                    primary_keys.append((name, StorageObj._conversions[value]))
                columns = []
                for val in dict_vals.split(","):
                    name, value = StorageObj._data_type.match(val).groups()
                    columns.append((name, StorageObj._conversions[value]))
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
        return this

    def init_prefetch(self, block):
        """
        Initializes the prefetch manager of the storageobj persistentdict
        Args:
           block (hecuba.iter.Block): the dataset partition which need to be prefetch
        """
        self._get_default_dict().init_prefetch(block)

    def end_prefetch(self):
        """
        Terminates the prefetch manager of the storageobj persistentdict
        """
        self._get_default_dict().end_prefetch()

    def getByName(self):
        """
        When running the StorageObj.__init__ function with parameters, it retrieves the persistent dict from the DDBB,
        by creating a PersistentDict which links to the Cassandra columnfamily
        """

        props = self.__class__._persistent_props
        dictionaries = filter(lambda (k, t): t['type'] == 'dict', props.iteritems())
        for table_name, per_dict in dictionaries:
            pd = PersistentDict(self._ksp, table_name, self._persistent, per_dict['primary_keys'], per_dict['columns'])
            setattr(self, table_name, pd)
            self._persistent_dicts.append(pd)

    def make_persistent(self):
        """
        Once a StorageObj has been created, it can be made persistent. This function retrieves the information about
        the Object class schema, and creates a Cassandra table with those parameters, where information will be saved
        from now on, until execution finishes or StorageObj is no longer persistent.
        It also inserts into the new table all information that was in memory assigned to the StorageObj prior to this
        call.
        """


        for dict_name, props in self.__class__._persistent_props.iteritems():
            columns = map(lambda a: '%s %s' % a, props['primary_keys'] + props['columns'])
            pks = map(lambda a: a[0], props['primary_keys'])
            querytable = "CREATE TABLE IF NOT EXISTS %s.%s (%s, PRIMARY KEY (%s));" % (self._ksp, dict_name,
                                                                                      str.join(",", columns),
                                                                                      str.join(',', pks))
            session.execute(querytable)

        create_attrs = "CREATE TABLE IF NOT EXISTS %s.%s ("% (self._ksp, self._table)+\
            "name text PRIMARY KEY, " + \
            "intval int, " + \
            "intlist list<int>, "+\
            "inttuple list<int>, "+\
            "doubleval double, " + \
            "doublelist list<double>, " +\
            "doubletuple list<double>, "+\
            "textval text, "+\
            "textlist list<text>, "+\
            "texttuple list<text>)"

        session.execute(create_attrs)
        self._persistent = True
        for dict in self._persistent_dicts:
            memory_vals = dict.iteritems()
            dict.is_persistent = True
            for key, val in memory_vals:
                dict[key] = val
        for key, variable in vars(self).iteritems():
            self.__setattr__(key, variable)



    def iteritems(self):
        """
        Calls the iterator for the keys of the storageobj
        Returns:
            self: list of key,val pairs
        """
        raise ValueError('not yet implemented')

    def itervalues(self):
        """
        Calls the iterator to obtain the values of the storageobj
        Returns:
            self: list of keys
        Currently blocked to avoid cache inconsistencies
        """
        print "Data should be accessed through a block"
        raise ValueError('not yet implemented')

    def increment(self, target, value):
        """
        Instead of increasing the existing value in position target, it sets it to the desired value (only intended to
        be used with counter tables
        """
        self[target] = value

    def _get_default_dict(self):
        '''

        Returns:
             PersistentDict: the first persistent dict
        '''
        if len(self._persistent_dicts) == 0:
            raise KeyError('There are no persistent dicts')
        return self._persistent_dicts[0]

    def empty_persistent(self):
        """
            Empties the Cassandra table where the persistent StorageObj stores data
        """
        def_dict = self._get_default_dict()
        def_dict.dictCache.cache = {}

        query = "TRUNCATE %s.%s;" % (self._ksp, self._table)
        session.execute(query)
        for d in self._persistent_dicts:
            query = "TRUNCATE %s.%s;" % (self._ksp, d._table)
            session.execute(query)

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
        session.execute(query)
        query = "DROP COLUMNFAMILY %s.%;" % (self._ksp, self._table)
        session.execute(query)

    def __contains__(self, key):
        """
           Returns True if the given key can be found in the PersistentDict, false otherwise
           Args:
               key: key that we are looking for in the PersistentDict
           Returns:
               a (boolean): True if the given key can be found in the PersistentDict, false otherwise
        """
        def_dict = self._get_default_dict()
        return def_dict.__contains__(key)

    def has_key(self, key):
        """
          Returns True if the given key can be found in the PersistentDict, false otherwise
          Args:
              key: key that we are looking for in the PersistentDict
          Returns:
              a (boolean): True if the given key can be found in the PersistentDict, false otherwise
        """
        def_dict = self._get_default_dict()
        def_dict.has_key(key)

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
            col_name = self._attr_to_column[key]
            query = "SELECT "+col_name+" FROM "+self._ksp+"."+self._table+" WHERE name = %s"
            result = session.execute(query, [key])
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
        if issubclass(PersistentDict, value.__class__):
            super(StorageObj, self).__setattr__(key, value)
        else:
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

                querytable = "INSERT INTO "+self._ksp+"."+self._table+"(name," + self._attr_to_column[key]+ \
                             ") VALUES (%s,%s)"


                session.execute(querytable, [key, value])

            else:
                super(StorageObj, self).__setattr__(key, value)

    def getID(self):
        """
        This function returns the ID of the StorageObj
        """
        return '%s_1' % self._myuuid

    def split(self):
        """
          Depending on if it's persistent or not, this function returns the list of keys of the PersistentDict assigned to
          the StorageObj, or the list of keys
          Returns:
               a) List of keys in case that the SO is not persistent
               b) Iterator that will return Blocks, one by one, where we can find the SO data in case it's persistent
        """

        if not self._persistent:
            auxdict = self._get_default_dict()
            return [auxdict.keys()]
        else:
            return KeyIter(self)

    def __additem__(self, key, other):
        """
           Depending on if it's persistent or not, this function adds the given value to the given key position:
               a) In memory
               b) In the DDBB
           Args:
               key: position of the value we want to increase
               other: quantity by which we want to increase the value stored in the key position
        """
        auxdict = self._get_default_dict()
        auxdict[key] += other

    def __getitem__(self, key):
        """
        Redirects the call to get the value at a given key position to the PersistentDict of the StorageObj
        Args:
            key:

        Returns:

        """
        auxdict = self._get_default_dict()
        item = auxdict[key]
        return item

    def __setitem__(self, key, value):
        auxdict = self._get_default_dict()
        auxdict[key] = value

    def statistics(self):

        keys = map(lambda a: a[0], self.__class__._persistent_props['primary_keys'])

        reads = 0
        exec ("reads = self." + keys[0] + ".reads")
        writes = 0
        exec ("writes = self." + keys[0] + ".writes")

        if reads > 0 or writes > 0:
            print "####################################################"
            print "STATISTICS"
            print "Object:", self.__class__.__name__
            print "----------------------------------------------------"

        if reads > 0:
            if reads < 10:
                print "reads:                         ", reads
            else:
                if reads < 100:
                    print "reads:                        ", reads
                else:
                    if reads < 1000:
                        print "reads:                       ", reads
                    else:
                        if reads < 10000:
                            print "reads:                      ", reads
                        else:
                            print "reads:                     ", reads

            chits = 0
            exec ("chits = self." + keys[0] + ".cache_hits")
            if chits < 10:
                print "cache_hits(X):                 ", chits
            else:
                if chits < 100:
                    print "cache_hits(X):                ", chits
                else:
                    if chits < 1000:
                        print "cache_hits(X):               ", chits
                    else:
                        if chits < 10000:
                            print "cache_hits(X):              ", chits
                        else:
                            print "cache_hits(X):             ", chits
            pendreqs = 0
            exec ("pendreqs = self." + keys[0] + ".pending_requests")
            if pendreqs > 0:
                if pendreqs < 10:
                    print "pending_reqs:                  ", pendreqs
                else:
                    if pendreqs < 100:
                        print "pending_reqs:                 ", pendreqs
                    else:
                        if pendreqs < 1000:
                            print "pending_reqs:                ", pendreqs
                        else:
                            if pendreqs < 10000:
                                print "pending_reqs:               ", pendreqs
                            else:
                                print "pending_reqs:              ", pendreqs
            dbhits = 0
            exec ("dbhits = self." + keys[0] + ".miss")
            if dbhits < 10:
                print "miss(_):                       ", dbhits
            else:
                if dbhits < 100:
                    print "miss(_):                      ", dbhits
                else:
                    if dbhits < 1000:
                        print "miss(_):                     ", dbhits
                    else:
                        if dbhits < 10000:
                            print "miss(_):                    ", dbhits
                        else:
                            print "miss(_):                   ", dbhits

            cprefetchs = 0
            exec ("cprefetchs = self." + keys[0] + ".cache_prefetchs")
            if cprefetchs > 0:
                if cprefetchs < 10:
                    print "cprefetchs:                    ", cprefetchs
                else:
                    if cprefetchs < 100:
                        print "cprefetchs:                   ", cprefetchs
                    else:
                        if cprefetchs < 1000:
                            print "cprefetchs:                  ", cprefetchs
                        else:
                            if cprefetchs < 10000:
                                print "cprefetchs:                 ", cprefetchs
                            else:
                                print "cprefetchs:                ", cprefetchs

            cachepreffails = 0
            exec ("cachepreffails = self." + keys[0] + ".cache_prefetchs_fails")
            if cachepreffails < 10:
                print "cachepreffails:                ", cachepreffails
            else:
                if cachepreffails < 100:
                    print "cachepreffails:                 ", cachepreffails
                else:
                    print "cachepreffails:                ", cachepreffails

            cache_usage = 0
            if reads > 0:
                cache_usage = (float(chits) / float(reads)) * 100
                if cache_usage < 10:
                    print("cache_usage(cache hits/reads):  %.2f%%" % cache_usage)
                else:
                    if cache_usage < 100:
                        print("cache_usage(cache hits/reads): %.2f%%" % cache_usage)
                    else:
                        print("cache_usage(cache hits/reads):%.2f%%" % cache_usage)
            '''
            if cprefetchs > 0:
                used_prefetchs = (float(chits) / float(cprefetchs)) * 100
                if used_prefetchs < 10:
                    print("used_prefetchs (cache hits/cprefetchs):    %.2f%%" % used_prefetchs)
                else:
                    if used_prefetchs < 100:
                        print("used_prefetchs (cache hits/cprefetchs):   %.2f%%" % used_prefetchs)
                    else:
                        print("used_prefetchs (cache hits/cprefetchs):  %.2f%%" % used_prefetchs)
            '''
            '''
            if reads > 0:
                pendreqstotal = (float(pendreqs) / float(reads)) * 100
                if pendreqstotal < 10:
                    print("pending reqs(pendreqs/reads):   %.2f%%" % pendreqstotal)
                else:
                    if pendreqstotal < 100:
                        print("pending reqs(pendreqs/reads):  %.2f%%" % pendreqstotal)
                    else:
                        print("pending reqs(pendreqs/reads): %.2f%%" % pendreqstotal)
            '''

        if writes > 0:
            if writes < 10:
                print "writes:                        ", writes
            else:
                if writes < 100:
                    print "writes:                       ", writes
                else:
                    if writes < 1000:
                        print "writes:                      ", writes
                    else:
                        if writes < 10000:
                            print "writes:                     ", writes
                        else:
                            if writes < 100000:
                                print "writes:                    ", writes
                            else:
                                print "writes:                   ", writes

            cachewrite = 0
            exec ("cachewrite = self." + str(keys[0]) + ".cachewrite")
            if cachewrite < 10:
                print "cachewrite:                    ", cachewrite
            else:
                if cachewrite < 100:
                    print "cachewrite:                   ", cachewrite
                else:
                    if cachewrite < 1000:
                        print "cachewrite:                  ", cachewrite
                    else:
                        if cachewrite < 10000:
                            print "cachewrite:                 ", cachewrite
                        else:
                            if cachewrite < 100000:
                                print "cachewrite:                ", cachewrite
                            else:
                                print "cachewrite:               ", cachewrite

            syncs = 0
            exec ("syncs = self." + str(keys[0]) + ".syncs")
            if syncs < 10:
                print "syncs:                         ", syncs
            else:
                if syncs < 100:
                    print "syncs:                        ", syncs
                else:
                    if syncs < 1000:
                        print "syncs:                       ", syncs
                    else:
                        if syncs < 10000:
                            print "syncs:                      ", syncs
                        else:
                            if syncs < 100000:
                                print "syncs:                     ", syncs
                            else:
                                print "syncs:                    ", syncs

        if reads > 0 or writes > 0:
            print "------Times-----------------------------------------"
        if reads > 0:
            print "GETS"
            exec ("cache_hits_time = self." + str(keys[0]) + ".cache_hits_time")
            print "cache_hits_time:               ", cache_hits_time
            cache_hits_time_med = 0.00000000000
            if chits > 0:
                cache_hits_time_med = cache_hits_time / chits
                print("cache_hits_time_med:            %.8f" % cache_hits_time_med)
            '''
            exec("pendreqsTimeRes = self." + str(keys[0]) + ".pending_requests_time_res")
            if pendreqsTimeRes < 10:
                print "pending_requests_time_res:     ", pendreqsTimeRes
            else:
                print "pending_requests_time_res:    ", pendreqsTimeRes
            pending_requests_time_med_res = 0.000000000000
            if pendreqs > 0:
                pending_requests_time_med_res = pendreqsTimeRes / pendreqs
                print("pending_requests_time_med_res:  %.8f" % pending_requests_time_med_res)
            '''
            '''
            exec("pendreqsTime = self." + str(keys[0]) + ".pending_requests_time")
            if pendreqsTime < 10:
                print "pending_requests_time:         ", pendreqsTime
            else:
                print "pending_requests_time:        ", pendreqsTime
            pending_requests_time_med = 0.000000000000
            if pendreqs > 0:
                pending_requests_time_med = pendreqsTime / pendreqs
                print("pending_requests_time_med:      %.8f" % pending_requests_time_med)
            '''
            '''
            pendreqfailstime = 0
            exec("pendreqfailstime = self." + str(keys[0]) + ".pending_requests_fails_time")
            if pendreqfailstime < 10:
                print "pendreqfailstime:              ", pendreqfailstime
            else:
                if pendreqfailstime < 100:
                    print "pendreqfailstime:               ", pendreqfailstime
                else:
                    print "pendreqfailstime:              ", pendreqfailstime
            '''
            exec ("mtime = self." + str(keys[0]) + ".miss_time")
            if mtime < 10:
                print "miss_time:                     ", mtime
            else:
                print "miss_time:                    ", mtime
            miss_times_med = 0.000000000000
            if dbhits > 0:
                miss_times_med = mtime / dbhits
                print("miss_times_med:                 %.8f" % miss_times_med)

            # print "total_read_time:               ", str(cache_hits_time + pendreqsTime + mtime)
            print "total_read_time:               ", str(cache_hits_time + mtime)

        if writes > 0:
            print "WRITES"

            exec ("syncstime = self." + str(keys[0]) + ".syncs_time")
            if syncstime < 10:
                print "syncs_time:                    ", syncstime
            else:
                print "syncs_time:                   ", syncstime
            syncs_times_med = 0.000000000000
            if syncs > 0:
                exec ("syncs_times_med = (self." + str(keys[0]) + ".syncs_time / syncs)")
            print("syncs_times_med:                %.8f" % syncs_times_med)

            exec ("cwritetime = self." + str(keys[0]) + ".cachewrite_time")
            if cwritetime < 10:
                print "cachewrite_time:               ", cwritetime
            else:
                print "cachewrite_time:              ", cwritetime
            cachewrite_times_med = 0.000000000000
            if cachewrite > 0:
                cachewrite_times_med = cwritetime / cachewrite
                print("cachewrite_times_med:           %.8f" % cachewrite_times_med)

            totalWritesTime = cwritetime + syncstime
            if totalWritesTime < 10:
                print "write_time:                    ", totalWritesTime
            else:
                print "write_time:                   ", totalWritesTime
            write_times_med = 0.00000000000
            if writes > 0:
                write_times_med = (totalWritesTime / writes)
                print("write_times_med:                %.8f" % write_times_med)

        if reads > 0:
            print "------Graph-----------------------------------------"
            exec ("print self." + str(keys[0]) + ".cache_hits_graph")
        if reads > 0 or writes > 0:
            print "####################################################"
