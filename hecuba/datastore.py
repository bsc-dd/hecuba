# author: G. Alomar
from hecuba.dict import *
from conf.apppath import apppath
import inspect
from pprint import pprint

def hecuba_filter(function, iterable):
    print "datastore hecuba_filter ####################################"
    print "function:         ", function
    print "iterable:         ", iterable
    print "type(iterable):   ", type(iterable)
    inspectedfunction = inspect.getsource(function)
    print "inspectedfunction:", inspectedfunction
    #pprint (vars(iterable))
    #if hasattr(iterable, 'indexed'):
    if iterable == []:
        print "indexed object"
        iterable.indexArguments = str(str(str(inspectedfunction).split(":")[1]).split(",")[0]).split(' and ')  # Args list
        print "iterable.indexArguments:", iterable.indexArguments
        return iterable
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
