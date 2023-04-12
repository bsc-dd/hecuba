
def producer_with_new_class():
    from DictWithStrings import DictWithStrings
    o = DictWithStrings("streaming_dict_with_str")

    o[666] = "Oh! Yeah! Holidays!";

    print("AFTER Setitem", flush=True)
    # No sync is done here, therefore the data is still in memory, but STREAM would send data anyway to the Consumer

def producer_with_new_classandNumpy():
    from DictWithNumpy import DictWithNumpy
    from hecuba import StorageNumpy
    import numpy as np
    o = DictWithNumpy("streaming_dict_with_numpy")
    n=np.arange(12).reshape(3,4)+1
    sn = StorageNumpy(n,"miclassNumpy")
    o[42]=sn
    print("AFTER Setitem", flush=True)
    # No sync is done here, therefore the data is still in memory, but STREAM would send data anyway to the Consumer

def producer_subclass_storageNumpy():
    from myNumpy import myNumpy
    import numpy as np

    x = myNumpy(np.arange(12,dtype=float).reshape(3,4)+1, "mynpsubclass")
    x.send()
    print("AFTER Send", flush=True)

def main():
    print("PRODUCER STARTING", flush=True)
    producer_with_new_classandNumpy()
    producer_with_new_class()
    #producer_subclass_storageNumpy()
    print("PRODUCER DONE", flush=True)



if __name__ == "__main__":
    main()
