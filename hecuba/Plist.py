from hecuba.iter import *
from hecuba.list import KeyList

class PersistentKeyList(KeyList):
    
    def __init__(self, mypdict):
        self.mypdict = mypdict
        

    def __iter__(self):
        print "PersistentKeyList __iter__ ####################################"
        return KeyIter(self)
        

    def getID(self):
	identifier = "%s_%s_kmeans_BlockData" % ( self.node, self.start_token )
	return identifier

