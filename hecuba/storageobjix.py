# author: G. Alomar
from hecuba.storageobj import StorageObj


class StorageObjIx(StorageObj):
    def split(self):
        #indexarguments = self.indexArguments
        #print "indexarguments:", indexarguments
        return super(StorageObjIx,self).split(self)
    
    #pass
