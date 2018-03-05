import unittest
from hecuba import config
from hecuba.hdict import StorageDict
from hecuba.storageobj import StorageObj
from hecuba.IStorage import IStorage, AlreadyPersistentError

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

class ExampleStorageObjClass2(StorageObj):
    '''
            @ClassField my_example int
            @ClassField my_example2 int
            @ClassField nombre1 tests.withcassandra.prueba.ExampleStorageObjClass
    '''


class TutorialTest(unittest.TestCase):

  


    def test1(self):
        tablename = 'examplestoragedictclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        my_example_class = ExampleStoragedictClass()
        my_example_class.make_persistent(tablename)
	my_example_class.delete_persistent()
	count, = config.session.execute('SELECT count(*) FROM my_app.'+tablename)[0]
	self.assertEqual(count,0)
	self.assertEqual(my_example_class.items(),[])


    def tests2(self):
        tablename = 'examplestoragedictclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        my_example_class = ExampleStoragedictClassInit()
        StorageDict.__init__(my_example_class)
        my_example_class.make_persistent(tablename)
	my_example_class.delete_persistent()
	count, = config.session.execute('SELECT count(*) FROM my_app.'+tablename)[0]
	self.assertEqual(count,0)
	self.assertEqual(my_example_class.items(),[])



    def tests3(self):
        tablename = 'examplestoragedictclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        my_example_class = ExampleStoragedictClassInitMultiVal()
        StorageDict.__init__(my_example_class)
        my_example_class.make_persistent(tablename)
	my_example_class.delete_persistent()
	count, = config.session.execute('SELECT count(*) FROM my_app.'+tablename)[0]
	self.assertEqual(count,0)
	self.assertEqual(my_example_class.items(),[])


	
    def tests4(self):
	tablename = 'examplestorageobjclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        my_example_class = ExampleStorageObjClass()
        my_example_class.make_persistent(tablename)
	my_example_class.delete_persistent()
	count, = config.session.execute('SELECT count(*) FROM my_app.'+tablename)[0]
	self.assertEqual(count,0)
	var=True
	for name in my_example_class._persistent_attrs:
		var= var & (name in my_example_class.__dict__.keys())
	self.assertEqual(var,False)


    def tests5(self):
	tablename = 'examplestorageobjclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename + '_my_dict')
        my_example_class = ExampleStorageObjClassInit()
        StorageObj.__init__(my_example_class)
        my_example_class.make_persistent(tablename)
	my_example_class.delete_persistent()
	count, = config.session.execute('SELECT count(*) FROM my_app.'+tablename)[0]
	self.assertEqual(count,0)
	var=True
	for name in my_example_class._persistent_attrs:
		var= var & (name in my_example_class.__dict__.keys())
	self.assertEqual(var,False)

    def tests6(self):
	tablename = 'examplestorageobjclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename + '_my_dict')
        my_example_class = ExampleStorageObjClassNames()
        my_example_class.make_persistent(tablename)
	my_example_class.delete_persistent()
	count, = config.session.execute('SELECT count(*) FROM my_app.'+tablename)[0]
	self.assertEqual(count,0)
	var=True
	for name in my_example_class._persistent_attrs:
		var= var & (name in my_example_class.__dict__.keys())
	self.assertEqual(var,False)

    def test7(self):
	tablename = 'examplestorageobjclass1'
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename + '_my_dict')
        my_example_class = ExampleStorageObjClass2()
        my_example_class.make_persistent(tablename)
	attr_istorage=[]
	for obj_name in my_example_class._persistent_attrs:
		attr= getattr(my_example_class,obj_name,None)
		if isinstance(attr,IStorage):
			attr_istorage.append(obj_name)
			
	my_example_class.delete_persistent()
	count, = config.session.execute('SELECT count(*) FROM my_app.'+tablename)[0]
	self.assertEqual(count,0)
	for obj_name in attr_istorage:
        	print 'my_app.'+tablename+'_'+obj_name	
              	count, = config.session.execute('SELECT count(*) FROM my_app.'+tablename+'_'+obj_name)[0]
                self.assertEqual(count,0)

if __name__ == '__main__':
    untest.main()
