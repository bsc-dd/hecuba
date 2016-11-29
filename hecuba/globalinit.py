from hecuba.dict import *
from hecuba.iter import *
from hecuba.storageobj import StorageObj
from hecuba.settings import config, session
import glob
import re

valid_type = '(atomicint|str|bool|decimal|float|int|tuple|list|generator|frozenset|set|dict|long|buffer|bytearray|counte)'
data_type = re.compile('(\w+) *: *%s' % valid_type)
cname = re.compile('.*class +(\w+) *\(StorageObj\):.*')
dict_case = re.compile('.*@ClassField +(\w+) +dict +< *< *([\w:]+) *> *, *([\w+:]+) *>.*')
val_case = re.compile('.*@ClassField +(\w+) +(\w+) +%s' % valid_type)
file_name = re.compile('.*(app/[^/]+)\.py')
conversion = {'atomicint': 'counter',
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

def classfilesparser():
    classes = {}
    files_to_parse = glob.glob(config.apppath + "/app/*.py")

    for ftp in files_to_parse:
        with open(ftp, 'r') as f:
            this = {'module_name': file_name.match(ftp).group(1).replace("/", "."), 'storage_objs': {}}
            for line in f:
                m = cname.match(line)
                if m is not None:
                    classes[m.groups()[0]] = this
                else:
                    m = dict_case.match(line)
                    if m is not None:
                        # Matching @ClassField of a dict
                        table_name, dict_keys, dict_vals = m.groups()
                        primary_keys = []
                        for key in dict_keys.split(","):
                            name, value = data_type.match(key).groups()
                            primary_keys.append((name, conversion[value]))
                        columns = []
                        for val in dict_vals.split(","):
                            name, value = data_type.match(val).groups()
                            columns.append((name, conversion[value]))
                        this['storage_objs'][table_name] = {
                            'type': 'dict',
                            'primary_keys': primary_keys,
                            'columns': columns}
                    else:
                        m = val_case.match(line)
                        if m is not None:
                            table_name, simple_type = m.groups()
                            this['storage_objs'][table_name] = {
                                'type': conversion[simple_type]
                            }
    return classes



__initialized = False

def hello_world():
    global __initialized
    if __initialized:
        return
    __initialized = True
    print "@@@@@@@@@@@@@@@@@@@ HELLO WORLD @@@@@@@@@@@@@@@@@@@@@@"

    keyspace = execution_name

    repl_factor = "3"
    repl_class = "SimpleStrategy"

    try:
        session.execute(
            "CREATE KEYSPACE IF NOT EXISTS " + keyspace + " WITH REPLICATION = { 'class' : \'" + repl_class + "\', 'replication_factor' : " + repl_factor + " };")
    except Exception as e:

        print "Cannot create keyspace", e

    KeyIter.blockkeyspace = keyspace
    PersistentDict.keyspace = keyspace

    classes1 = classfilesparser()
    for class_name, props in classes1.iteritems():
        exec 'from %s import %s' % (props['module_name'], class_name)
        exec 'so_class = ' + class_name
        so_class._persistent_props = props



