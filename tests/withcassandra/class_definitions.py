from hecuba import StorageObj, StorageDict

class SObj_Basic(StorageObj):
    '''
    @ClassField attr1 int
    @ClassField attr2 double
    @ClassField attr3 str
    '''


class SDict_SimpleTypeSpec(StorageDict):
    '''
    @TypeSpec <<id:int>,info:str>
    '''


class SDict_ComplexTypeSpec(StorageDict):
    '''
    @TypeSpec <<id:int>,state:tests.withcassandra.class_definitions.SObj_Basic>
    '''


class SObj_SimpleClassField(StorageObj):
    '''
    @ClassField attr1 int
    @ClassField mydict dict <<key:str>,value:double>
    @ClassField attr3 double
    '''


class SObj_ComplexClassField(StorageObj):
    '''
    @ClassField attr1 int
    @ClassField mydict dict <<key:str>,state:tests.withcassandra.class_definitions.SObj_Basic>
    @ClassField attr3 double
    '''




class MyStorageDict(StorageDict):
    '''
    @TypeSpec <<position:int>,val:int>
    '''
    pass


class MyStorageDict2(StorageDict):
    '''
    @TypeSpec <<position:int, position2:str>,val:int>
    '''
    pass


class MyStorageDict3(StorageDict):
    '''
    @TypeSpec <<str>,int>
    '''


class MyStorageObjC(StorageObj):
    '''
    @ClassField mona dict<<a:str>,b:int>
    '''


class MyStorageDictA(StorageDict):
    '''
    @TypeSpec <<a:str>,b:int>
    '''


class Words(StorageObj):
    '''
    @ClassField words dict<<position:int>,wordinfo:str>
    '''
    pass


class TestSimple(StorageObj):
    '''
    @ClassField words dict<<position:int>,value:str>
    '''
    pass



class Result(StorageObj):
    '''
    @ClassField instances dict<<word:str>,numinstances:int>
    '''
    pass


class TestStorageObj(StorageObj):
    '''
       @ClassField test dict<<position:int>,text:str>
    '''
    pass


class TestStorageIndexedArgsObj(StorageObj):
    '''
       @ClassField test dict<<position:int>,x:float,y:float,z:float>
       @Index_on test x,y,z
    '''
    pass





class Test2StorageObj(StorageObj):
    '''
       @ClassField name str
       @ClassField age int
    '''
    pass


class Test2StorageObjFloat(StorageObj):
    '''
       @ClassField name str
       @ClassField age float
    '''
    pass


class Test3StorageObj(StorageObj):
    '''
       @ClassField myso tests.withcassandra.class_definitions.Test2StorageObj
       @ClassField myso2 tests.withcassandra.class_definitions.TestStorageObj
       @ClassField myint int
       @ClassField mystr str
    '''
    pass


class Test4StorageObj(StorageObj):
    '''
       @ClassField myotherso tests.withcassandra.class_definitions.Test2StorageObj
    '''
    pass


class Test4bStorageObj(StorageObj):
    '''
       @ClassField myotherso tests.withcassandra.class_definitions.Test2StorageObj
    '''
    pass


class Test5StorageObj(StorageObj):
    '''
       @ClassField test2 dict<<position:int>,myso:tests.withcassandra.class_definitions.Test2StorageObj>
    '''
    pass


class Test6StorageObj(StorageObj):
    '''
       @ClassField test3 dict<<int>,str,str>
    '''
    pass


class Test7StorageObj(StorageObj):
    '''
       @ClassField test2 dict<<int>,tests.withcassandra.class_definitions.Test2StorageObj>
    '''
    pass


class TestStorageObjNumpy(StorageObj):
    '''
       @ClassField mynumpy numpy.ndarray
    '''
    pass


class TestStorageObjNumpyDict(StorageObj):
    '''
       @ClassField mynumpydict dict<<int>,numpy.ndarray>
    '''
    pass



class TestAttributes(StorageObj):
    '''
       @ClassField key int
    '''

    value = None

    def do_nothing_at_all(self):
        pass

    def setvalue(self, v):
        self.value = v

    def getvalue(self):
        return self.value


class mixObj(StorageObj):
    '''
    @ClassField floatfield float
    @ClassField intField int
    @ClassField strField str
    @ClassField intlistField list <int>
    @ClassField floatlistField list <float>
    @ClassField strlistField list <str>
    @ClassField dictField dict <<int>,str>
    @ClassField inttupleField tuple <int, int>
    '''
