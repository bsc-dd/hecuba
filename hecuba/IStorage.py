import uuid
from collections import namedtuple
from time import time
from hecuba import config, log


class IStorage:
    _select_istorage_meta = config.session.prepare("SELECT * FROM hecuba.istorage WHERE storage_id = ?")
    args_names = []
    args = namedtuple("IStorage", [])
    _build_args = args()

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
    def build_remotely(new_args):
        raise Exception("to be implemented")

    def split(self):
        st = time()
        tokens = self._build_args.tokens

        for token_split in IStorage._tokens_partitions(tokens, config.min_number_of_tokens, config.number_of_blocks):
            storage_id = str(uuid.uuid1())
            log.debug('assigning to %s tokens %s', storage_id, token_split)
            new_args = self._build_args._replace(tokens=token_split, storage_id=storage_id)
            self.__class__._store_meta(new_args)

            yield self.__class__.build_remotely(new_args)
        log.debug('completed split of %s in %f', self.__class__.__name__, time()-st)

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
                yield tokens[len(tokens)/splits * splits + 1:]

    @staticmethod
    def _build_discrete_token_ranges():
        """
        Makes proper tokens ranges ensuring that in a tuple (a,b) a <= b

        :param tokens:  a list of tksn [1, 0,10]
        :return:  a rationalized list [(-1, 0),(0,10),(10, max)]
        """
        token_to_host_owner = config.cluster.metadata.token_map.token_to_host_owner
        from cassandra.metadata import Murmur3Token

        start = Murmur3Token(-2 ** 63)
        end = Murmur3Token(-1 + 2 ** 63)
        last_token = max(token_to_host_owner.keys())
        if start not in token_to_host_owner:
            token_to_host_owner[start] = token_to_host_owner[last_token]

        if end not in token_to_host_owner:
            token_to_host_owner[end] = token_to_host_owner[last_token]

        tokens = token_to_host_owner.keys()
        tokens.sort()
        n_tns = len(tokens)
        token_ranges = []
        for i in range(0, n_tns - 1):
            token_ranges.append((tokens[i], tokens[i + 1]))

        token_ranges.sort(key=lambda (_, a): token_to_host_owner[a].address)

        return map(lambda (mm3_from, mm3_to): (mm3_from.value, mm3_to.value), token_ranges)

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
        return (ksp, table)

    def make_persistent(self, name):
        raise Exception("to be implemented")

    def stop_persistent(self):
        raise Exception("to be implemented")

    def delete_persistent(self):
        raise Exception("to be implemented")

    def getID(self):
        return self._storage_id
