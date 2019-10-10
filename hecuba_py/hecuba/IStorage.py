import re
from bisect import bisect_right
from collections import namedtuple, defaultdict

from hecuba import config
from hecuba.partitioner import partitioner_split


class AlreadyPersistentError(RuntimeError):
    pass


_select_istorage_meta = config.session.prepare("SELECT * FROM hecuba.istorage WHERE storage_id = ?")
_size_estimates = config.session.prepare(("SELECT mean_partition_size, partitions_count "
                                          "FROM system.size_estimates WHERE keyspace_name=? and table_name=?"))
_max_token = int(((2 ** 63) - 1))  # type: int
_min_token = int(-2 ** 63)  # type: int

_valid_types = ['counter', 'text', 'boolean', 'decimal', 'double', 'int', 'list', 'set', 'map', 'bigint', 'blob',
                'tuple', 'dict', 'float', 'numpy.ndarray']

_basic_types = _valid_types[:-1]
_hecuba_valid_types = '(atomicint|str|bool|decimal|float|int|tuple|list|generator|frozenset|set|dict|long|buffer' \
                      '|counter|double)'

AT = 'int | atomicint | str | bool | decimal | float | long | double | buffer'

ATD = 'int | atomicint | str | bool | decimal | float | long | double | buffer | set'

_python_types = [int, str, bool, float, tuple, set, dict, bytearray]
_conversions = {'atomicint': 'counter',
                'str': 'text',
                'bool': 'boolean',
                'decimal': 'decimal',
                'float': 'float',
                'int': 'int',
                'tuple': 'tuple',
                'list': 'list',
                'generator': 'list',
                'frozenset': 'set',
                'set': 'set',
                'dict': 'map',
                'long': 'bigint',
                'buffer': 'blob',
                'bytearray': 'blob',
                'counter': 'counter',
                'double': 'double',
                'StorageDict': 'dict',
                'ndarray': 'hecuba.hnumpy.StorageNumpy',
                'numpy.ndarray': 'hecuba.hnumpy.StorageNumpy'}

args = namedtuple("IStorage", [])


def process_path(module_path):
    """
    Method to obtain module and class_name from a module path
    Args:
        module_path(String): path in the format module.class_name
    Returns:
        tuple containing class_name and module
    """

    if module_path == 'numpy.ndarray':
        return 'StorageNumpy', 'hecuba.hnumpy'
    if module_path == 'StorageDict':
        return 'StorageDict', 'hecuba.hdict'
    last = 0
    for key, i in enumerate(module_path):
        if i == '.' and key > last:
            last = key
    module = module_path[:last]
    class_name = module_path[last + 1:]
    return class_name, module


def _discrete_token_ranges(tokens):
    """
    Makes proper tokens ranges ensuring that in a tuple (a,b) a <= b
    Args:
        tokens:  a list of tokens [1, 0, 10]
    Returns:
         a rationalized list [(-1, 0),(0,10),(10, max)]
    """
    tokens.sort()
    if len(tokens) == 0:
        return tokens
    if tokens[0] > _min_token:
        token_ranges = [(_min_token, tokens[0])]
    else:
        token_ranges = []
    n_tns = len(tokens)
    for i in range(0, n_tns - 1):
        token_ranges.append((tokens[i], tokens[i + 1]))
    token_ranges.append((tokens[n_tns - 1], _max_token))
    return token_ranges


def _extract_ks_tab(name):
    """
    Method used to obtain keyspace and table from a given name
    Args:
        name: a string containing keyspace name and table name, or only table name
    Returns:
        a tuple containing keyspace name and table name
    """
    sp = name.split(".")
    if len(sp) == 2:
        ksp = sp[0]
        table = sp[1]
    else:
        ksp = config.execution_name
        table = name
    return ksp.lower(), table.lower()


class IStorage:
    args_names = []
    _build_args = args()
    _storage_id = None

    def split(self):
        """
        Method used to divide an object into sub-objects.
        Returns:
            a subobject everytime is called
        """
        return partitioner_split(self)

    def _get_istorage_attrs(self, storage_id):
        return list(config.session.execute(_select_istorage_meta, [storage_id]))

    def _count_name_collision(self, attribute):
        m = re.compile("^%s_%s(_[0-9]+)?$" % (self._table.lower(), attribute))
        q = config.session.execute("SELECT table_name FROM  system_schema.tables WHERE keyspace_name = %s",
                                   [self._ksp])
        return sum(1 for elem in q if m.match(elem[0]))

    @staticmethod
    def build_remotely(args):
        """
        Takes the information which consists of at least the type,
        :raises TypeError if the object class doesn't subclass IStorage
        :param obj_info: Contains the information to be used to create the IStorage obj
        :return: An IStorage object
        """
        if "built_remotely" not in args.keys():
            built_remotely = True
        else:
            built_remotely = args["built_remotely"]

        obj_type = args.get('class_name', args.get('type', None))
        if obj_type is None:
            raise TypeError("Trying to build an IStorage obj without giving the type")

        # Import the class defined by obj_type
        cname, module = process_path(obj_type)

        try:
            mod = __import__(module, globals(), locals(), [cname], 0)
        except ValueError:
            raise ValueError("Can't import class {} from module {}".format(cname, module))

        imported_class = getattr(mod, cname)
        if not issubclass(imported_class, IStorage):
            raise TypeError("Trying to build remotely an object '%s' != IStorage subclass" % cname)

        args = {k: v for k, v in args.items() if k in imported_class.args_names}
        args.pop('class_name', None)
        args["built_remotely"] = built_remotely

        return imported_class(**args)

    @staticmethod
    def _store_meta(storage_args):
        raise Exception("to be implemented")

    def make_persistent(self, name):
        raise Exception("to be implemented")

    def stop_persistent(self):
        raise Exception("to be implemented")

    def delete_persistent(self):
        raise Exception("to be implemented")

    def getID(self):
        """
        Obtains the id of the storage element
        Returns:
            self._storage_id: id of the object
        """
        return str(self._storage_id)
