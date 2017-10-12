from hecuba.hnumpy import *
import uuid
ud = uuid.uuid4()
a = StorageNumpy(np.zeros([100,100]),ud)
a.make_persistent('test.crashit')