import uuid
from collections import namedtuple

from hecuba import config


class IStorage:
    args_names = []
    args = namedtuple("IStorage")
    _build_args = args()

    @staticmethod
    def build_remotely(results):
        pass

    def split(self):
        tokens = self._build_args.tokens
        splits = max(len(tokens) / config.number_of_blocks, 1)

        for i in range(0, len(tokens), splits):
            myuuid = str(uuid.uuid1())
            new_args = self._build_args._replace(tokens=tokens[i:i + splits], storage_id=myuuid)
            args = ','.join(self.args_names)
            place_holders = ','.join(map(lambda a: '%s', self.args_names))
            query = 'INSERT INTO hecuba.istorage(%s) VALUES (%s)' % (args, place_holders)
            print query
            config.session.execute(query, list(new_args))

            yield self.__class__.build_remotely(new_args)

    def make_persistent(self):
        pass

    def stop_persistent(self):
        pass

    def delete_persistent(self):
        pass

    def getID(self):
        pass
