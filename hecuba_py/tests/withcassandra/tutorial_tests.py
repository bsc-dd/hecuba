import unittest
from hecuba import config
from hecuba.hdict import StorageDict
from hecuba.storageobj import StorageObj


class ExampleStoragedictClass(StorageDict):
    '''
        @TypeSpec <<int>, str>
    '''


class ExampleStoragedictClassInit(StorageDict):
    '''
        @TypeSpec <<int>, str>
    '''
    def __init__(self,**kwargs):
        super(ExampleStoragedictClassInit, self).__init__(**kwargs)
        self.update({0: 'first position'})


class ExampleStoragedictClassInitMultiVal(StorageDict):
    '''
        @TypeSpec <<int, int>, str, str, int>
    '''
    def __init__(self,**kwargs):
        super(ExampleStoragedictClassInitMultiVal, self).__init__(**kwargs)
        self.update({(0, 1):('first position', 'second_position', 1000)})


class ExampleStoragedictClassNames(StorageDict):
    '''
        @TypeSpec <<position:int>, value:str>
    '''


class ExampleStoragedictClassInitNames(StorageDict):
    '''
        @TypeSpec <<position:int>, value:str>
    '''
    def __init__(self,**kwargs):
        super(ExampleStoragedictClassInitNames, self).__init__(**kwargs)
        self.update({0: 'first position'})


class ExampleStoragedictClassInitMultiValNames(StorageDict):
    '''
        @TypeSpec <<position1:int, position2:int>, value1:str, value2:str, value3:int>
    '''
    def __init__(self,**kwargs):
        super(ExampleStoragedictClassInitMultiValNames, self).__init__(**kwargs)
        self.update({(0, 1): ('first position', 'second_position', 1000)})


class ExampleStorageObjClass(StorageObj):
    '''
        @ClassField my_dict dict<<int>, str>
        @ClassField my_release int
        @ClassField my_version string
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


class ExampleStorageObjClassNames(StorageObj):
    '''
        @ClassField my_dict dict<<position:int>, str>
        @ClassField my_release int
        @ClassField my_version string
    '''


class TutorialTest(unittest.TestCase):

    def init_storagedict_test(self):
        tablename = 'examplestoragedictclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        my_example_class = ExampleStoragedictClass()
        my_example_class.make_persistent(tablename)

    def init_storagedictwithinit_test(self):
        tablename = 'examplestoragedictclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        my_example_class = ExampleStoragedictClassInit()
        StorageDict.__init__(my_example_class)
        my_example_class.make_persistent(tablename)

    def init_storagedictwithinitmultival_test(self):
        tablename = 'examplestoragedictclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        my_example_class = ExampleStoragedictClassInitMultiVal()
        StorageDict.__init__(my_example_class)
        my_example_class.make_persistent(tablename)

    def init_storagedictnames_test(self):
        tablename = 'examplestoragedictclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        my_example_class = ExampleStoragedictClassNames()
        my_example_class.make_persistent(tablename)

    def init_storagedictwithinitnames_test(self):
        tablename = 'examplestoragedictclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        my_example_class = ExampleStoragedictClassInitNames()
        StorageDict.__init__(my_example_class)
        my_example_class.make_persistent(tablename)

    def init_storagedictwithinitmultivalnames_test(self):
        tablename = 'examplestoragedictclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        my_example_class = ExampleStoragedictClassInitMultiValNames()
        StorageDict.__init__(my_example_class)
        my_example_class.make_persistent(tablename)

    def init_storageobj_test(self):
        tablename = 'examplestorageobjclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename + '_my_dict')
        my_example_class = ExampleStorageObjClass()
        my_example_class.make_persistent(tablename)

    def init_storageobjwithinit_test(self):
        tablename = 'examplestorageobjclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename + '_my_dict')
        my_example_class = ExampleStorageObjClassInit()
        StorageObj.__init__(my_example_class)
        my_example_class.make_persistent(tablename)

    def init_storageobjnames_test(self):
        tablename = 'examplestorageobjclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename + '_my_dict')
        my_example_class = ExampleStorageObjClassNames()
        my_example_class.make_persistent(tablename)

    def initializestoragedict_test(self):
        tablename = 'examplestorageobjclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename + '_my_dict')
        my_persistent_storageDict = StorageDict(tablename, [('position', 'int')], [('value', 'text')])
