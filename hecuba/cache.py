# author: G. Alomar
class PersistentDictCache(dict):
    cache = {}
    pending_requests = {}
    sents = 0
    valType = ""
    keyType = ""

    def __init__(self):
        self.cache = {}
        self.pending_requests = {}
        self.sents = 0
        self.valType = ""
        self.keyType = ""

    def __setitem__(self, key, value):
        self.cache[key] = value
              
    def __getitem__(self, key):
        value = ''
        try:
            value = self.cache[key]
        except Exception as e:
            print "Error when retrieving value from cache:", e
            print "self.cache:", self.cache
        return value
