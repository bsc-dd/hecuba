from hecuba.iter import KeyIter

class KeyList(list):
    
    def __init__(self, mypdict):
        self.mypdict = mypdict
        

    def __iter__(self):
        return KeyIter(self)
    
