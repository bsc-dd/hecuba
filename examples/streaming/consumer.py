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


def main():
    print("CONSUMER STARTING", flush=True)
    consumer_with_new_classandNumpy()
    consumer_with_new_class()
    print("CONSUMER DONE", flush=True)



if __name__ == "__main__":
    main()
