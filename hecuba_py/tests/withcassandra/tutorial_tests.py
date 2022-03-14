import unittest
from hecuba import config
from hecuba.hdict import StorageDict
from hecuba.storageobj import StorageObj


class ExampleStoragedictClass(StorageDict):
    '''
        @TypeSpec dict<<k1:int>, val1:str>
    '''


class ExampleStoragedictClassInit(StorageDict):
    '''
        @TypeSpec dict<<k1:int>, val1:str>
    '''

    def __init__(self, **kwargs):
        super(ExampleStoragedictClassInit, self).__init__(**kwargs)
        self.update({0: 'first position'})


class ExampleStoragedictClassInitMultiVal(StorageDict):
    '''
        @TypeSpec dict<<k1:int, k2:int>, val1:str, val2:str, val3:int>
    '''

    def __init__(self, **kwargs):
        super(ExampleStoragedictClassInitMultiVal, self).__init__(**kwargs)
        self.update({(0, 1): ['first position', 'second_position', 1000]})


class ExampleStoragedictClassNames(StorageDict):
    '''
        @TypeSpec dict<<position:int>, value:str>
    '''


class ExampleStoragedictClassInitNames(StorageDict):
    '''
        @TypeSpec dict<<position:int>, value:str>
    '''

    def __init__(self, **kwargs):
        super(ExampleStoragedictClassInitNames, self).__init__(**kwargs)
        self.update({0: 'first position'})


class ExampleStoragedictClassInitMultiValNames(StorageDict):
    '''
        @TypeSpec dict<<position1:int, position2:int>, value1:str, value2:str, value3:int>
    '''

    def __init__(self, **kwargs):
        super(ExampleStoragedictClassInitMultiValNames, self).__init__(**kwargs)
        self.update({(0, 1): ['first position', 'second_position', 1000]})


class ExampleStorageObjClass(StorageObj):
    '''
        @ClassField my_dict dict<<k1:int>, val1:str>
        @ClassField my_release int
        @ClassField my_version str
    '''


class ExampleStorageObjClassInit(StorageObj):
    '''
        @ClassField my_dict dict<<k1:int>, val1:str>
        @ClassField my_release int
        @ClassField my_version str
    '''

    def __init__(self, **kwargs):
        super(ExampleStorageObjClassInit, self).__init__(**kwargs)
        self.my_dict = {0: 'first position'}
        self.my_release = 2017
        self.my_version = '0.1'


class ExampleStorageObjClassNames(StorageObj):
    '''
        @ClassField my_dict dict<<position:int>, val1:str>
        @ClassField my_release int
        @ClassField my_version str
    '''


class TutorialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = config.execution_name
        config.NUM_TEST = 0 # HACK a new attribute to have a global counter
    @classmethod
    def tearDownClass(cls):
        config.execution_name = cls.old
        del config.NUM_TEST

    # Create a new keyspace per test
    def setUp(self):
        config.NUM_TEST = config.NUM_TEST + 1
        self.current_ksp = "TutorialTest{}".format(config.NUM_TEST).lower()
        config.execution_name = self.current_ksp

    def tearDown(self):
        #config.session.execute("DROP KEYSPACE IF EXISTS {}".format(self.current_ksp))
        pass

    def test_init_storagedict_test(self):
        tablename = 'examplestoragedictclass1'
        my_example_class = ExampleStoragedictClass()
        my_example_class.make_persistent(tablename)

    def test_init_storagedictwithinit_test(self):
        tablename = 'examplestoragedictclass1'
        my_example_class = ExampleStoragedictClassInit()
        StorageDict.__init__(my_example_class)
        my_example_class.make_persistent(tablename)

    def test_init_storagedictwithinitmultival_test(self):
        tablename = 'examplestoragedictclass1'
        my_example_class = ExampleStoragedictClassInitMultiVal()
        StorageDict.__init__(my_example_class)
        my_example_class.make_persistent(tablename)

    def test_init_storagedictnames_test(self):
        tablename = 'examplestoragedictclass1'
        my_example_class = ExampleStoragedictClassNames()
        my_example_class.make_persistent(tablename)

    def test_init_storagedictwithinitnames_test(self):
        tablename = 'examplestoragedictclass1'
        my_example_class = ExampleStoragedictClassInitNames()
        StorageDict.__init__(my_example_class)
        my_example_class.make_persistent(tablename)

    def test_init_storagedictwithinitmultivalnames_test(self):
        tablename = 'examplestoragedictclass1'
        my_example_class = ExampleStoragedictClassInitMultiValNames()
        StorageDict.__init__(my_example_class)
        my_example_class.make_persistent(tablename)

    def test_init_storageobj_test(self):
        tablename = 'examplestorageobjclass1'
        my_example_class = ExampleStorageObjClass()
        my_example_class.make_persistent(tablename)

    def test_init_storageobjwithinit_test(self):
        tablename = 'examplestorageobjclass1'
        my_example_class = ExampleStorageObjClassInit()
        StorageObj.__init__(my_example_class)
        my_example_class.make_persistent(tablename)

    def test_init_storageobjnames_test(self):
        tablename = 'examplestorageobjclass1'
        my_example_class = ExampleStorageObjClassNames()
        my_example_class.make_persistent(tablename)

    def test_initializestoragedict_test(self):
        tablename = 'examplestorageobjclass1'
        my_persistent_storageDict = StorageDict(tablename, [('position', 'int')], [('value', 'text')])
