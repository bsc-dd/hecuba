import uuid
from collections import namedtuple
from time import time
from hecuba import config, log
import re


class IStorage:
    _select_istorage_meta = config.session.prepare("SELECT * FROM hecuba.istorage WHERE storage_id = ?")
    args_names = []
    args = namedtuple("IStorage", [])
    _build_args = args()

    _valid_types = ['counter', 'text', 'boolean', 'decimal', 'double', 'int', 'list', 'set', 'map', 'bigint', 'blob',
                    'counter', 'dict', 'float']

    _hecuba_valid_types = '(atomicint|str|bool|decimal|float|int|tuple|list|generator|frozenset|set|dict|long|buffer' \
                          '|numpy.ndarray|counter)'
    _data_type = re.compile('(\w+) *: *%s' % _hecuba_valid_types)
    _so_data_type = re.compile('(\w+)*:(\w.+)')
    _list_case = re.compile('.*@ClassField +(\w+) +list+ *< *([\w:.+]+) *>')
    _sub_dict_case = re.compile(' *< *< *([\w:, ]+)+ *> *, *([\w+:, <>]+) *>')
    _sub_tuple_case = re.compile(' *< *([\w:, ]+)+ *>')
    _val_case = re.compile('.*@ClassField +(\w+) +%s' % _hecuba_valid_types)
    _so_val_case = re.compile('.*@ClassField +(\w+) +([\w.]+)')

    _python_types = [int, str, bool, float, tuple, set, dict, long, bytearray]

    _conversions = {'atomicint': 'counter',
                    'str': 'text',
                    'bool': 'boolean',
                    'decimal': 'decimal',
                    'float': 'double',
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
                    'counter': 'counter'}

    @staticmethod
    def process_path(module_path):
        last = 0
        for key, i in enumerate(module_path):
            if i == '.' and key > last:
                last = key
        module = module_path[:last]
        cname = module_path[last + 1:]
        return cname, module

    @staticmethod
    def build_remotely(new_args):
        raise Exception("to be implemented")

    def split(self):
        st = time()
        tokens = self._build_args.tokens

        for token_split in IStorage._tokens_partitions(tokens, config.min_number_of_tokens, config.number_of_blocks):
            storage_id = uuid.uuid4()
            log.debug('assigning to %s %d  tokens', str(storage_id), len(token_split))
            new_args = self._build_args._replace(tokens=token_split, storage_id=storage_id)
            self.__class__._store_meta(new_args)

            yield self.__class__.build_remotely(new_args)
        log.debug('completed split of %s in %f', self.__class__.__name__, time() - st)

    @staticmethod
    def _tokens_partitions(tokens, min_number_of_tokens, number_of_blocks):
        if len(tokens) < min_number_of_tokens:
            # In this case we have few token and thus we split them
            tkns_for_block = min_number_of_tokens / number_of_blocks
            step_size = ((2 ** 64) - 1) / min_number_of_tokens
            block = []
            for fraction, to in tokens:
                while fraction < to - step_size:
                    block.append((fraction, fraction + step_size))
                    fraction += step_size
                    if len(block) >= tkns_for_block:
                        yield block
                        block = []
                # Adding the last token

                block.append((fraction, to))
            if len(block) > 0:
                yield block
        else:
            # This is the case we have more tokens than blocks,.
            splits = max(len(tokens) / number_of_blocks, 1)

            for i in xrange(0, len(tokens), splits):
                yield tokens[i:i + splits]
            if len(tokens) % splits > 0:
                yield tokens[len(tokens) / splits * splits + 1:]

    @staticmethod
    def _discrete_token_ranges(tokens):
        """
        Makes proper tokens ranges ensuring that in a tuple (a,b) a <= b
        :param tokens:  a list of tksn [1, 0,10]
        :return:  a rationalized list [(-1, 0),(0,10),(10, max)]
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
    def _store_meta(storage_args):
        raise Exception("to be implemented")

    @staticmethod
    def _extract_ks_tab(name):
        sp = name.split(".")
        if len(sp) == 2:
            ksp = sp[0]
            table = sp[1]
        else:
            ksp = config.execution_name
            table = name
        return ksp.lower().encode('UTF8'), table.lower().encode('UTF8')

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
            self._storage_id: id of the block
        """
        # raise Exception("to be implemented")
        return str(self._storage_id)
