import numpy as np
from mynumpyclass import mynumpyclass

def producer_singleNumpy_int():
    n = np.arange(12,dtype=int).reshape(3,4) +1
    mysn=mynumpyclass(n,"myintnumpy")
    mysn.send()

def producer_singleNumpy_float():
    n = np.arange(12,dtype=float).reshape(3,4) +1
    mysn=mynumpyclass(n,"myfloatnumpy")
    mysn.send()

if __name__ == "__main__":
    producer_singleNumpy_int()
    producer_singleNumpy_float()
