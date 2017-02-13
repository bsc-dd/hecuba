import uuid
from collections import namedtuple

from hecuba import config


class IStorage:
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
    def build_remotely(storage_id):
        raise Exception("to be implemented")

    def split(self):
        tokens = self._build_args.tokens
        splits = max(len(tokens) / config.number_of_blocks, 1)

        for i in range(0, len(tokens), splits):
            storage_id = str(uuid.uuid1())
            new_args = self._build_args._replace(tokens=tokens[i:i + splits], storage_id=storage_id)
            self.__class__._store_meta(new_args)

            yield self.__class__.build_remotely(storage_id)
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
        return self.storage_id
