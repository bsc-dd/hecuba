# author: G. Alomar
from hecuba.storageobj import StorageObj
from hecuba.Plist import *


class StorageObjIx(StorageObj):
    def split(self):
        #indexarguments = self.indexArguments
        #print "indexarguments:", indexarguments
        #return super(StorageObjIx,self).split()
        keys = self.keyList[self.__class__.__name__]
        if not self.persistent:
            exec("a = dict.keys(self." + str(keys[0]) + ")")
            return a
        else:
            exec("a = IxPersistentKeyList(self." + str(keys[0]) + ")")
            return a
    
    #pass
