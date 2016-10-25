class AtomicInteger():

        value = 0
        def __init__ (self,value):
                self.value = value


        def increment(self, value):
                self.value = value
                return self.value
