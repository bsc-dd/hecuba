# author: G. Alomar
from hecuba.storageobj import StorageObj


class StorageObjIx(StorageObj):
    def split(self):
        indexarguments = self.indexArguments
        print "indexarguments:", indexarguments
