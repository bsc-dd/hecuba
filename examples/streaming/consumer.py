#from model_hecuba_stream import miclass
from hecuba_stream import miclassNumpy

def consumer_with_new_class():
    from hecuba_stream import miclass
    o = miclass("streaming_dict_with_str")
    k,v = o.poll()
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
    print("=========================", flush=True)

def consumer_with_new_classandNumpy():
    import numpy as np

    o = miclassNumpy("streaming_dict_with_numpy")
    k,v=o.poll()
    print("AFTER POLL", flush=True)
    print("key ", k, flush=True)
    print("value ", v, flush=True)
    print("o[key] ", o[k], flush=True) # it should be already in memory (without any Cassandra Access)
    passed=True
    if k != 42:
        passed = False
    if not np.array_equal(v, (np.arange(12,dtype=float).reshape(4,3)+1)):
        passed = False
    if not passed:
        print("consumer_with_new_classandNumpy NOT PASSED", flush=True)
    else:
        print("consumer_with_new_classandNumpy PASSED", flush=True)
    print("=========================", flush=True)


def consumer_subclass_storageNumpy():
    from hecuba_stream import myNumpy
    import numpy as np

    # The following code instantiates the 'myNumpy' class, but it may happen that the
    # producer has not been executed yet, and therefore ... the object does not
    # exist yet, ... so we keep retrying until it exists. The problem is that
    # there is no difference between this case and the instantiation of an
    # already created object.
    exist = False
    while not exist:
        try:
            x = myNumpy(None, "i_am_a_numpy")
            exist = True
        except ValueError:
            pass

    v = x.poll()

    if not np.array_equal(v, (np.arange(12,dtype=float).reshape(4,3)+1)):
        print("consumer_subclass_storageNumpy NOT PASSED", flush=True)
    else:
        print("consumer_subclass_storageNumpy PASSED", flush=True)
    print("=========================", flush=True)



def main():
    print("CONSUMER STARTING", flush=True)
    consumer_with_new_classandNumpy()
    consumer_with_new_class()
    consumer_subclass_storageNumpy()
    print("CONSUMER DONE", flush=True)



if __name__ == "__main__":
    main()
