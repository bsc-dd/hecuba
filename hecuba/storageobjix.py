# author: G. Alomar
from hecuba.storageobj import StorageObj
from hecuba.Plist import *


class StorageObjIx(StorageObj):

    def __init__(self, name=None):
        print "storageobj __init__ ####################################"
        print self.__class__.__name__
        setattr(self, 'name', None)
        setattr(self, 'persistent', False)
        setattr(self, 'indexed', False)
        setattr(self, 'indexArguments', '')
        #self.getByName(name)
        super(StorageObjIx, self).getByName(name)

    def split(self):
        print "StorageObjIx split ####################################"
        #indexarguments = self.indexArguments
        #print "indexarguments:", indexarguments
        #return super(StorageObjIx,self).split()
        keys = self.keyList[self.__class__.__name__]
        print "keys:", keys
        if not self.persistent:
            print "not self.persistent"
            exec("a = dict.keys(self." + str(keys[0]) + ")")
            return a
        else:
            print "yes self.persistent"
            exec("a = IxPersistentKeyList(self." + str(keys[0]) + ")")
            return a
    
    #pass
