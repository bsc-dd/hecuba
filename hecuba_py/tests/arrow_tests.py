import numpy as np
from hecuba import StorageNumpy

x = np.arange(12).reshape(3,4)

s = StorageNumpy(x, "hola")

