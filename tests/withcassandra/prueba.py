import unittest
from hecuba import config
from hecuba.hdict import StorageDict
from hecuba.storageobj import StorageObj

# coding: utf-8
class ExampleStorageObjClass(StorageObj):
    '''
        @ClassField my_example int
        @ClassField my_example2 int
    '''


class ExampleStorageObjClassInit(StorageObj):
    '''
        @ClassField my_dict dict<<int>, str>
        @ClassField my_release int
        @ClassField my_version string
    '''
    def __init__(self,**kwargs):
        super(ExampleStorageObjClassInit, self).__init__(**kwargs)
        self.my_dict = {0: 'first position'}
        self.my_release = 2017
        self.my_version = '0.1'
        self.perro="hola"


class ExampleStorageObjClassNames(StorageObj):
    '''
        @ClassField my_dict dict<<position:int>, str>
        @ClassField my_release int
        @ClassField my_version string
    '''

class ExampleStorageObjClass2(StorageObj):
    '''
        @ClassField my_example int
        @ClassField my_example2 int
        @ClassField nombre1 ExampleStorageObjClass
    '''
    
