import numpy as np
from mynumpyclass import mynumpyclass

def consumer_singleNumpy_int():
    mysn=mynumpyclass.get_by_alias("myintnumpy")
    mysn.poll()
    print(mysn)

def consumer_singleNumpy_float():
    mysn=mynumpyclass.get_by_alias("myfloatnumpy")
    mysn.poll()
    print(mysn)

if __name__ == "__main__":
    consumer_singleNumpy_int()
    consumer_singleNumpy_float()
