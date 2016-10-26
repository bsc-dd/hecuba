# author: G. Alomar
from hecuba.dict import *
from conf.apppath import apppath
import inspect
from pprint import pprint

def hecuba_filter(function, iterable):
    print "iterable:", iterable
    print iterable[0].__class__.__name__
    print iterable[1].__class__.__name__
    pprint (vars(iterable[0]))
    if hasattr(iterable[0], 'indexed'):
        print "indexed object"
        inspectedfunction = inspect.getsource(function)
        iterable[1].indexArguments = str(str(str(inspectedfunction).split(":")[1]).split(",")[0]).split(' and ')  # Args list
        print "iterable.indexArguments:", iterable[1].indexArguments
        return iterable[1]
    else:
        print "normal object"
        filtered = python_filter(function, iterable)
        return filtered

path = apppath + '/conf/storage_params.txt'

file = open(path, 'r')

for line in file:
    exec line

if not filter == hecuba_filter:
    python_filter = filter
    filter = hecuba_filter
