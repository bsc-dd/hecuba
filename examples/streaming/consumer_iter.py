##########
# NOTICE:
# The classes 'DictWithStrings' and 'DictWithNumpy' required in this file
# are generated automatically by 'producer.cpp' and are like the following:
#
# DictWithNumpy.py:
# ================
#from hecuba import StorageDict, StorageStream
#
#class DictWithNumpy(StorageDict, StorageStream):
#   '''
#   @TypeSpec dict <<keyname0:int>,valuename0:hecuba.hnumpy.StorageNumpy>
#   '''
# DictWithStrings.py:
# ===================
#from hecuba import StorageDict, StorageStream
#
#class DictWithStrings(StorageDict, StorageStream):
#   '''
#   @TypeSpec dict <<keyname0:int>,valuename0:str>
#   '''
#
# If 'producer.cpp' has not been executed in the current directory you need to
# create these files manually.



def consumer_with_new_class():
    from DictWithStrings import DictWithStrings
    o = DictWithStrings("streaming_dict_with_str")
    for k,v in o.items():
        print("AFTER POLL", flush=True)
        print("key ", k, flush=True)
        print("value ", v, flush=True)
        passed=True
        if k != 666:
            passed = False
        if v != "Oh! Yeah! Holidays!":
            passed = False
        if not passed:
            print("consumer_with_new_class NOT PASSED", flush=True)
        else:
            print("consumer_with_new_class PASSED", flush=True)
    print("consumer_with_new_class === FINISHED", flush=True)

def consumer_with_new_classandNumpy():
    from DictWithNumpy import DictWithNumpy
    import numpy as np

    o = DictWithNumpy("streaming_dict_with_numpy")
    for k,v in o.items():
        print("AFTER POLL", flush=True)
        print("key ", k, flush=True)
        print("value ", v, flush=True)
        print("o[key] ", o[k], flush=True) # it should be already in memory (without any Cassandra Access)
        passed=True
        if k != 42:
            passed = False
        if not np.array_equal(v, (np.arange(12,dtype=float).reshape(3,4)+1)):
            passed = False
        if not passed:
            print("consumer_with_new_classandNumpy NOT PASSED", flush=True)
        else:
            print("consumer_with_new_classandNumpy PASSED", flush=True)
        print("=========================", flush=True)
    print("consumer_with_new_classandNumpy ===FINISHED", flush=True)

def consumer_subclass_storageNumpy():
    from myNumpy import myNumpy
    import numpy as np

    # The following code instantiates the 'myNumpy' class, but it may happen that the
    # producer has not been executed yet, and therefore ... the object does not
    # exist yet, ... so we keep retrying until it exists. The problem is that
    # there is no difference between this case and the instantiation of an
    # already created object. Change that to get_by_alias
    exist = False
    while not exist:
        try:
            #x = myNumpy(None, "mynpsubclass")
            x = myNumpy.getByAlias("mynpsubclass")
            exist = True
        except ValueError:
            pass

    v = x.poll()

    if not np.array_equal(v, (np.arange(12,dtype=float).reshape(3, 4)+1)):
        print("consumer_subclass_storageNumpy NOT PASSED", flush=True)
        print("Expected: {} ".format(np.arange(12, dtype=float).reshape(3,4)+1), flush=True)
        print("Received: {} ".format(v), flush=True)
    else:
        print("consumer_subclass_storageNumpy PASSED", flush=True)
    print("=========================", flush=True)


def main():
    print("CONSUMER STARTING", flush=True)
    consumer_with_new_classandNumpy()
    consumer_with_new_class()
    #consumer_subclass_storageNumpy()
    print("CONSUMER DONE", flush=True)



if __name__ == "__main__":
    main()
