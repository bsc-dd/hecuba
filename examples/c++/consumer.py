#from model_hecuba_stream import miclass
from hecuba_stream import miclassNumpy

def consumer_with_new_class():
    from hecuba_stream import miclass
    o = miclass("streaming_dict_with_str")
    k,v = o.poll()
    print("AFTER POLL", flush=True)
    print("key ", k, flush=True)
    print("value ", v, flush=True)

def consumer_with_new_classandNumpy():
    o = miclassNumpy("streaming_dict_with_numpy")
    k,v=o.poll()
    print("AFTER POLL", flush=True)
    print("key ", k, flush=True)
    print("value ", v, flush=True)
    print("o[key] ", o[k], flush=True)


def main():
    print("CONSUMER STARTING", flush=True)
    consumer_with_new_classandNumpy()
    consumer_with_new_class()
    print("CONSUMER DONE", flush=True)



if __name__ == "__main__":
    main()
