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
    print("BEFORE INSTATIATION", flush=True)
    from DictWithStrings import DictWithStrings
    o = DictWithStrings("streaming_dict_with_str")
    print("AFTER INSTATIATION", flush=True)
    k,v = o.poll()
    print("AFTER POLL", flush=True)
    print("key ", k, flush=True)
    print("value ", v, flush=True)
    passed=True
    if k != 666:
        passed = False
    if v != "Oh! Yeah! Holidays!":
        passed = False
    # Using 'poll' implies an EXTRA poll for the end of dictionary (EOD)
    k,v=o.poll()
    if k is not None:
        passed = False
    if not passed:
        print("consumer_with_new_class NOT PASSED", flush=True)
    else:
        print("consumer_with_new_class PASSED", flush=True)
    print("=========================", flush=True)

def consumer_with_new_classandNumpy():
    from DictWithNumpy import DictWithNumpy
    import numpy as np

    o = DictWithNumpy("streaming_dict_with_numpy")
    k,v=o.poll()
    print("AFTER POLL", flush=True)
    print("key ", k, flush=True)
    print("value ", v, flush=True)
    print("o[key] ", o[k], flush=True) # it should be already in memory (without any Cassandra Access)
    passed=True
    if k != 42:
        passed = False
    if not np.array_equal(v, (np.arange(12,dtype=float).reshape(3,4)+1)):
        passed = False
    # Using 'poll' implies an EXTRA poll for the end of dictionary (EOD)
    k,v=o.poll()
    if k is not None:
        passed = False
    if not passed:
        print("consumer_with_new_classandNumpy NOT PASSED", flush=True)
    else:
        print("consumer_with_new_classandNumpy PASSED", flush=True)
    print("=========================", flush=True)

def consumer_with_multiple_values():
    from DictWithMultValue import DictWithMultValue
    import numpy as np

    o = DictWithMultValue("streaming_dict_with_multiplevalues")
    passed=True
    for k,v1,v2 in o.items():
        print("AFTER POLL", flush=True)
        print("key ", k, flush=True)
        print("value1 ", v1, flush=True)
        print("value2 ", v2, flush=True)
        print("o[key] ", o[k], flush=True) # it should be already in memory (without any Cassandra Access)
        if k != 42:
            passed = False
            break
        if v1 != 43:
            passed = False
            break
        if not np.array_equal(v2, (np.arange(12,dtype=float).reshape(3,4)+1)):
            passed = False
            break
    if not passed:
        print("consumer_with_multiple_values NOT PASSED", flush=True)
    else:
        print("consumer_with_multiple_values PASSED", flush=True)
    print("=========================", flush=True)

def consumer_with_multiple_values2():
    from DictWithMultValue2 import DictWithMultValue2
    import numpy as np

    o = DictWithMultValue2("streaming_dict_with_multiplevalues2")
    passed=True
    #k,v1,v2=o.poll()
    for k,v1,v2 in o.items():
    #if True:
        print("AFTER POLL", flush=True)
        print("key ", k, flush=True)
        print("value1 ", v1, flush=True)
        print("value2 ", v2, flush=True)
        print("o[key] ", o[k], flush=True) # it should be already in memory (without any Cassandra Access)
        if k != 42:
            passed = False
            break
        if not np.array_equal(v1, (np.arange(12,dtype=float).reshape(3,4)+1)):
            passed = False
            break
        if v2 != 43:
            passed = False
            break
    if not passed:
        print("consumer_with_multiple_values2 NOT PASSED", flush=True)
    else:
        print("consumer_with_multiple_values2 PASSED", flush=True)
    print("=========================", flush=True)


def consumer_subclass_storageNumpy():
    from myNumpy import myNumpy
    import numpy as np

    # The following code instantiates the 'myNumpy' class, but it may happen that the
    # producer has not been executed yet, and therefore ... the object does not
    # exist yet, ... so we keep retrying until it exists. The problem is that
    # there is no difference between this case and the instantiation of an
    # already created object.
    exist = False
    while not exist:
        try:
            x = myNumpy(None, "mynpsubclass")
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

def consumer_with_multiple_basic_values():
    from DictWithMultipleBasicTypes import DictWithMultipleBasicTypes

    o = DictWithMultipleBasicTypes("streaming_dict_with_multibasicvalues")
    k,v1,v2=o.poll()
    passed=True
    #for k,v in o.items():
    if True:
        print("AFTER POLL", flush=True)
        print("key ", k, flush=True)
        print("value1 ", v1, flush=True)
        print("value2 ", v2, flush=True)
        print("o[key] ", o[k], flush=True) # it should be already in memory (without any Cassandra Access)
        if k != 666:
            passed = False
            #break
        if v1 != 42:
            passed = False
            #break
        if v2 != "Oh! Yeah! Holidays!":
            passed = False
            #break
    # Using 'poll' implies an EXTRA poll for the end of dictionary (EOD)
    k,v1,v2=o.poll()
    if k is not None:
        passed = False
    if not passed:
        print("consumer_with_multiple_basic_values NOT PASSED", flush=True)
    else:
        print("consumer_with_multiple_basic_values PASSED", flush=True)
    print("=========================", flush=True)



def main():
    consumer_with_new_classandNumpy()
    consumer_with_multiple_basic_values()
    consumer_with_multiple_values()
    consumer_with_multiple_values2()
    consumer_with_new_class()
    #consumer_subclass_storageNumpy()



if __name__ == "__main__":
    main()
