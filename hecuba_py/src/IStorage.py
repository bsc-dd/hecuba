import uuid
from collections import namedtuple
from time import time
from hecuba import config, log
import re
import regex


class AlreadyPersistentError(RuntimeError):
    pass


class IStorage:
    _select_istorage_meta = config.session.prepare("SELECT * FROM hecuba.istorage WHERE storage_id = ?")
    args_names = []
    args = namedtuple("IStorage", [])
    _build_args = args()

    _valid_types = ['counter', 'text', 'boolean', 'decimal', 'double', 'int', 'list', 'set', 'map', 'bigint', 'blob',
                    'tuple', 'dict', 'float', 'numpy.ndarray']

    _basic_types = _valid_types[:-1]
    _hecuba_valid_types = '(atomicint|str|bool|decimal|float|int|tuple|list|generator|frozenset|set|dict|long|buffer' \
                          '|counter|double)'

    AT = 'int | atomicint | str | bool | decimal | float | long | double | buffer'

    ATD = 'int | atomicint | str | bool | decimal | float | long | double | buffer | set'

    _python_types = [int, str, bool, float, tuple, set, dict, long, bytearray]
    _storage_id = None
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

    @staticmethod
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

    def split(self):
        """
        Method used to divide an object into sub-objects.
        Returns:
            a subobject everytime is called
        """
        st = time()
        tokens = self._build_args.tokens

        for token_split in IStorage._tokens_partitions(tokens, config.min_number_of_tokens,
                                                       config.number_of_partitions):
            storage_id = uuid.uuid4()
            log.debug('assigning to %s %d  tokens', str(storage_id), len(token_split))
            new_args = self._build_args._replace(tokens=token_split, storage_id=storage_id)
            yield self.__class__.build_remotely(new_args)
        log.debug('completed split of %s in %f', self.__class__.__name__, time() - st)

    @staticmethod
    def _tokens_partitions(tokens, min_number_of_tokens, number_of_partitions):
        """
        Method that calculates the new token partitions for a given object
        Args:
            tokens: current number of tokens of the object
            min_number_of_tokens: defined minimum number of tokens
            number_of_partitions: defined
        Returns:
            a partition everytime it's called
        """
        if len(tokens) < min_number_of_tokens:
            # In this case we have few token and thus we split them
            tkns_per_partition = min_number_of_tokens / number_of_partitions
            step_size = ((2 ** 64) - 1) / min_number_of_tokens
            partition = []
            for fraction, to in tokens:
                while fraction < to - step_size:
                    partition.append((fraction, fraction + step_size))
                    fraction += step_size
                    if len(partition) >= tkns_per_partition:
                        yield partition
                        partition = []
                # Adding the last token
                partition.append((fraction, to))
            if len(partition) > 0:
                yield partition
        else:
            # This is the case we have more tokens than partitions,.
            splits = max(len(tokens) / number_of_partitions, 1)

            for i in xrange(0, len(tokens), splits):
                yield tokens[i:i + splits]

    @staticmethod
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
        if tokens[0] > -2 ** 63:
            token_ranges = [(-2 ** 63, tokens[0])]
        else:
            token_ranges = []
        n_tns = len(tokens)
        for i in range(0, n_tns - 1):
            token_ranges.append((tokens[i], tokens[i + 1]))
        token_ranges.append((tokens[n_tns - 1], (2 ** 63) - 1))
        return token_ranges

    @staticmethod
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
        return ksp.lower().encode('UTF8'), table.lower().encode('UTF8')

    def _get_istorage_attrs(self, storage_id):
        return list(config.session.execute(self._select_istorage_meta, [storage_id]))

    def _count_name_collision(self, attribute):
        m = re.compile("^%s_%s(_[0-9]+)?$" % (self._table, attribute))
        q = config.session.execute("SELECT table_name FROM  system_schema.tables WHERE keyspace_name = %s",
                                   [self._ksp])
        return len(filter(lambda (t_name, ): m.match(t_name), q))

    def _build_istorage_obj(self, **obj_info):
        """
        Takes the information which consists of at least the type,
        :raises TypeError if the object class doesn't subclass IStorage
        :param obj_info: Contains the information to be used to create the IStorage obj
        :return: An IStorage object
        """
        try:
            obj_type = obj_info['type']
        except KeyError:
            raise TypeError("Trying to build an IStorage obj without giving the type")

        obj_info['class_name'] = obj_type

        # Import the class defined by obj_type
        cname, module = IStorage.process_path(obj_type)

        try:
            mod = __import__(module, globals(), locals(), [cname], 0)
        except ValueError:
            raise ValueError("Can't import class {} from module {}".format(cname, module))

        is_class = getattr(mod, cname)
        if not issubclass(is_class, IStorage):
            raise TypeError("Trying to build remotely an object '%s' != IStorage subclass" % cname)

        # Build the object's namedtuple from the given arguments
        namedtuple_args = [obj_info.get(arg, None) for arg in is_class.args_names]
        obj_namedtuple = is_class.args(*namedtuple_args)
        # Build the IStorage object through build_remotely method
        return is_class.build_remotely(obj_namedtuple)

    @staticmethod
    def build_remotely(new_args):
        raise Exception("to be implemented")

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
