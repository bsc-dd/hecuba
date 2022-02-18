import time
import unittest
import uuid
import datetime

import cassandra
import numpy as np
from hecuba import config
from hecuba.tools import discrete_token_ranges
from hecuba.storageobj import StorageObj
from storage.api import getByID
from hecuba.IStorage import build_remotely

from ..app.words import Words


class Test2StorageObj(StorageObj):
    '''
       @ClassField name str
       @ClassField age int
    '''
    pass


class Result(StorageObj):
    '''
    @ClassField instances dict<<word:str>, numinstances:int>
    '''
    pass


class TestStorageObj(StorageObj):
    '''
       @ClassField test dict<<position:int>, text:str>
    '''
    pass


class TestStorageIndexedArgsObj(StorageObj):
    '''
       @ClassField test dict<<position:int>, x:float, y:float, z:float>
       @Index_on test x,y,z
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
       @ClassField myso tests.withcassandra.storageobj_tests.Test2StorageObj
       @ClassField myso2 tests.withcassandra.storageobj_tests.TestStorageObj
       @ClassField myint int
       @ClassField mystr str
    '''
    pass


class Test4StorageObj(StorageObj):
    '''
       @ClassField myotherso tests.withcassandra.storageobj_tests.Test2StorageObj
    '''
    pass


class Test4bStorageObj(StorageObj):
    '''
       @ClassField myotherso tests.withcassandra.test2storageobj.Test2StorageObj
    '''
    pass


class Test5StorageObj(StorageObj):
    '''
       @ClassField test2 dict<<position:int>, myso:tests.withcassandra.storageobj_tests.Test2StorageObj>
    '''
    pass


class Test6StorageObj(StorageObj):
    '''
       @ClassField test3 dict<<key0:int>, val0:str, val1:str>
    '''
    pass


class Test7StorageObj(StorageObj):
    '''
       @ClassField test2 dict<<key0:int>, val0:tests.withcassandra.storageobj_tests.Test2StorageObj>
    '''
    pass


class TestStorageObjNumpy(StorageObj):
    '''
       @ClassField mynumpy numpy.ndarray
    '''
    pass


class TestStorageObjNumpyDict(StorageObj):
    '''
       @ClassField mynumpydict dict<<key:int>, val:numpy.ndarray>
    '''
    pass

class TestStorageObjDict(StorageObj):
    '''
        @ClassField MyAttribute_1 int
        @ClassField MyAttribute_2 dict <<int>, str>
        @ClassField MyAttribute_3 dict <<int, str>, int>
    '''


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
    @ClassField intlistField list<int>
    @ClassField floatlistField list<float>
    @ClassField strlistField list<str>
    @ClassField dictField dict<<key0:int>, val0:str>
    @ClassField inttupleField tuple<int,int>
    '''


class TestDate(StorageObj):
    '''
    @ClassField attr date
    '''


class TestTime(StorageObj):
    '''
    @ClassField attr time
    '''


class TestDateTime(StorageObj):
    '''
    @ClassField attr datetime
    '''



class StorageObjTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = config.execution_name
        config.execution_name = "StorageObjTest".lower()

    @classmethod
    def tearDownClass(cls):
        #config.session.execute("DROP KEYSPACE IF EXISTS {}".format(config.execution_name), timeout=60)
        config.execution_name = cls.old

    def setUp(self):
        self.current_ksp = config.execution_name
        pass

    def tearDown(self):
        pass

    def test_build_remotely(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObj")
        time.sleep(1)
        tablename="test_build_remotely"
        obj = TestStorageObj(config.execution_name + "." + tablename)
        r = {"built_remotely": False, "storage_id": uuid.uuid5(uuid.NAMESPACE_DNS, config.execution_name + '.' + tablename),
             "ksp": config.execution_name,
             "class_name": str(TestStorageObj.__module__) + "." + TestStorageObj.__name__, "name": tablename,
             "columns": [('val1', 'str')], "entry_point": 'localhost', "primary_keys": [('pk1', 'int')],
             "istorage_props": {},
             "tokens": discrete_token_ranges([token.value for token in config.cluster.metadata.token_map.ring])}

        nopars = build_remotely(r)
        self.assertEqual('TestStorageObj'.lower(), nopars._table)
        self.assertEqual(config.execution_name, nopars._ksp)
        self.assertEqual(uuid.uuid5(uuid.NAMESPACE_DNS, config.execution_name + '.' + tablename), nopars.storage_id)
        name, tkns = \
            config.session.execute("SELECT name, tokens FROM hecuba.istorage WHERE storage_id = %s",
                                   [nopars.storage_id])[
                0]

        self.assertEqual(name, config.execution_name + '.' + tablename)
        self.assertEqual(tkns, r['tokens'])
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObj")

    def test_init_create_pdict(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Result")
        time.sleep(1)

        tablename = 'test_init_create_pdict'
        r = {"built_remotely": False, "storage_id": uuid.uuid5(uuid.NAMESPACE_DNS, config.execution_name + '.' + tablename),
             "ksp": config.execution_name,
             "class_name": str(TestStorageObj.__module__) + "." + TestStorageObj.__name__, "name": tablename,
             "columns": [('val1', 'str')], "entry_point": 'localhost', "primary_keys": [('pk1', 'int')],
             "istorage_props": {},
             "tokens": discrete_token_ranges([token.value for token in config.cluster.metadata.token_map.ring])}

        nopars = build_remotely(r)
        self.assertEqual(nopars._built_remotely, False)
        self.assertEqual('TestStorageObj'.lower(), nopars._table)
        self.assertEqual(config.execution_name, nopars._ksp)
        self.assertEqual(uuid.uuid5(uuid.NAMESPACE_DNS, config.execution_name + '.' + tablename), nopars.storage_id)
        name, tkns = \
            config.session.execute("SELECT name,tokens FROM hecuba.istorage WHERE storage_id = %s",
                                   [nopars.storage_id])[0]
        self.assertEqual(name, config.execution_name + '.' + r['name'])
        self.assertEqual(tkns, r['tokens'])

        tkns = discrete_token_ranges(
            [8508619251581300691, 8514581128764531689, 8577968535836399533, 8596162846302799189,
             8603491526474728284, 8628291680139169981, 8687301163739303017, 9111581078517061776])
        tablename2 = tablename+'2'
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + '.' + tablename2)
        nopars = Result(name=tablename2,
                        tokens=tkns)
        self.assertEqual('Result'.lower(), nopars._table)
        self.assertEqual(config.execution_name, nopars._ksp)
        self.assertEqual(uuid.uuid5(uuid.NAMESPACE_DNS, config.execution_name + '.' + tablename2), nopars.storage_id)
        self.assertEqual(True, nopars._is_persistent)
        self.assertTrue(hasattr(nopars, 'instances'))
        name, read_tkns = config.session.execute("SELECT name,tokens FROM hecuba.istorage WHERE storage_id = %s",
                                                 [nopars.storage_id])[0]

        self.assertEqual(name, config.execution_name + '.' + tablename2)
        self.assertEqual(tkns, read_tkns)

    def test_mixed_class(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".mixObj")
        time.sleep(1)
        myObj = mixObj()

        myObj.make_persistent("test_mixed_class")

        myObj.floatfield = 5.0
        myObj.intField = 5
        myObj.strField = "6"
        myObj.intlistField = [7, 8, 9]
        myObj.floatlistField = [10.0, 11.0, 12.0]
        myObj.strlistField = ["13.0", "14.0", "15.0"]
        myObj.inttupleField = (1, 2)

        floatfield, intField, strField, intlistField, floatlistField, strlistField, inttupleField = \
            config.session.execute("SELECT floatField, "
                                   "intField, "
                                   "strField, "
                                   "intlistField, "
                                   "floatlistField, "
                                   "strlistField, "
                                   "inttupleField "
                                   "FROM " + self.current_ksp + ".mixObj WHERE storage_id =" + str(myObj.storage_id))[0]

        self.assertEquals(floatfield, myObj.floatfield)
        self.assertEquals(intField, myObj.intField)
        self.assertEquals(strField, myObj.strField)
        self.assertEquals(intlistField, myObj.intlistField)
        self.assertEquals(floatlistField, myObj.floatlistField)
        self.assertEquals(strlistField, myObj.strlistField)
        self.assertEquals(inttupleField, myObj.inttupleField)

    def test_init_empty(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObj")
        time.sleep(1)
        nopars = TestStorageObj('{}.test_init_empty'.format(self.current_ksp))
        self.assertEqual('TestStorageObj'.lower(), nopars._table)
        self.assertEqual(self.current_ksp,  nopars._ksp)

        res = config.session.execute(
            'SELECT storage_id, class_name, name, tokens, istorage_props FROM hecuba.istorage WHERE storage_id = %s',
            [nopars.storage_id])[0]

        storage_id, storageobj_classname, name, tokens, istorage_props = res
        self.assertEqual(storage_id, nopars.storage_id)
        self.assertEqual(storageobj_classname, TestStorageObj.__module__ + "." + TestStorageObj.__name__)
        self.assertEqual(name, '{}.test_init_empty'.format(self.current_ksp))

        rebuild = build_remotely(res._asdict())
        self.assertEqual(rebuild._built_remotely, True)
        self.assertEqual('TestStorageObj'.lower(), rebuild._table)
        self.assertEqual(self.current_ksp.lower(), rebuild._ksp)
        self.assertEqual(storage_id, rebuild.storage_id)

        self.assertEqual(nopars._is_persistent, rebuild._is_persistent)
        # self.assertEqual(vars(nopars), vars(rebuild))

    def test_make_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Words")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test6StorageObj")
        time.sleep(1)
        nopars = Words()
        self.assertFalse(nopars._is_persistent)
        nopars.ciao = 1
        nopars.ciao2 = "1"
        nopars.ciao3 = [1, 2, 3]
        nopars.ciao4 = (1, 2, 3)
        for i in range(10):
            nopars.words[i] = 'ciao' + str(i)

        #count, = config.session.execute(
        #    "SELECT count(*) FROM system_schema.tables WHERE keyspace_name = '"+self.current_ksp+"' and table_name = 'words'")[0]
        #self.assertEqual(0, count)

        nopars.make_persistent(self.current_ksp+".test_make_persistentsso")
        tablename = nopars.words._table
        del nopars

        count, = config.session.execute('SELECT count(*) FROM '+self.current_ksp +"."+ tablename)[0]
        self.assertEqual(10, count)

        nopars2 = Test6StorageObj(self.current_ksp+".test_make_persistentnonames")
        nopars2.test3[0] = ['1', '2']
        tablename = nopars2.test3._table
        time.sleep(2)
        result = config.session.execute("SELECT val0, val1 FROM "+self.current_ksp +"."+ tablename+" WHERE key0 = 0")

        rval0 = None
        rval1 = None
        for row in result:
            rval0 = row.val0
            rval1 = row.val1

        self.assertEqual('1', rval0)
        self.assertEqual('2', rval1)

    def test_empty_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Words")
        time.sleep(1)
        so = Words()
        so.make_persistent("test_empty_persistent")
        so.ciao = "an attribute"
        so.another = 123
        config.batch_size = 1
        config.cache_activated = False
        for i in range(10):
            so.words[i] = str.join(',', map(lambda a: "ciao", range(i)))

        table_name = so.words._name
        del so
        import gc
        gc.collect()

        # The attribute words has been persisted...
        count, = config.session.execute("SELECT COUNT(*) FROM "+self.current_ksp+".Words")[0]
        self.assertEqual(count, 1)
        # The 10 dict items have been persisted...
        count, = config.session.execute('SELECT count(*) FROM '+table_name)[0]
        self.assertEqual(10, count)
        #so = Words() FIXME: This is not possible anymore as the table_name for the attribute is randomly generated
        #so.make_persistent(self.current_ksp+".test_empty_persistent")
        #so.delete_persistent()
        #count, = config.session.execute('SELECT count(*) FROM '+self.current_ksp+'.test_empty_persistent_words')[0]
        #self.assertEqual(0, count)

    def test_simple_stores_after_make_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        so = Test2StorageObj()
        so.name = 'caio'
        so.age = 1000
        so.make_persistent("test_simple_stores_after_make_persistent")
        count, = config.session.execute("SELECT COUNT(*) FROM "+self.current_ksp+".Test2StorageObj")[0]
        self.assertEqual(count, 1)
        self.assertEqual(so.name, 'caio')
        self.assertEqual(so.age, 1000)

    def test_simple_attributes(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        so = Test2StorageObj()
        so.make_persistent("test_simple_attributes")
        so.name = 'caio'
        so.age = 1000
        count, = config.session.execute("SELECT COUNT(*) FROM "+self.current_ksp+".Test2StorageObj")[0]
        self.assertEqual(count, 1)
        self.assertEqual(so.name, 'caio')
        self.assertEqual(so.age, 1000)

    def test_modify_simple_attributes(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        so = Test2StorageObj()
        so.make_persistent("test_modify_simple_attributes")
        so.name = 'caio'
        so.age = 1000
        count, = config.session.execute("SELECT COUNT(*) FROM "+self.current_ksp+".Test2StorageObj")[0]
        self.assertEqual(count, 1)
        self.assertEqual(so.name, 'caio')
        self.assertEqual(so.age, 1000)
        so.name = 'addio'
        so.age = 2000
        self.assertEqual(so.name, 'addio')
        self.assertEqual(so.age, 2000)

    def test_delattr_nonpersistent(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        so = Test2StorageObj()
        so.name = 'caio'
        del so.name

        def del_attr():
            my_val = so.name

        self.assertRaises(AttributeError, del_attr)

    def test_delattr_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        so = Test2StorageObj("test_delattr_persistent")
        so.name = 'caio'
        del so.name

        def del_attr1():
            my_val = so.name

        self.assertRaises(AttributeError, del_attr1)

        def del_attr2():
            my_val = so.random_val

        self.assertRaises(AttributeError, del_attr1)

    def test_delattr_persistent_nested(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test3StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObj")
        time.sleep(1)
        so = Test3StorageObj("test_delattr_persistent_nested")
        nestedSo = Test2StorageObj()
        nestedSo.name = 'caio'
        so.myint = 123
        so.myso = nestedSo
        # Make sure the inner object has been made persistent
        self.assertTrue(nestedSo._is_persistent)
        # Delete the attribute
        del so.myint

        def del_attr1():
            my_val = so.myint

        # Accessing deleted attr of type StorageOb should raise AttrErr
        self.assertRaises(AttributeError, del_attr1)

        # We assign again, nestedSo still existed (no one called delete on it)
        so.myso = nestedSo

        # Delete a nested attribute of the shared StorageObj
        del so.myso.name

        # Make sure that the nested attribute deleted has been successfully deleted from both objects
        def del_attr2():
            my_val = nestedSo.name

        def del_attr3():
            my_val = so.myso.name

        self.assertRaises(AttributeError, del_attr2)
        self.assertRaises(AttributeError, del_attr3)

    def test_modify_simple_before_mkp_attributes(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        so = Test2StorageObj()
        so.name = 'caio'
        so.age = 1000
        so.make_persistent("test_modify_simple_before_mkp_attributes")
        count, = config.session.execute("SELECT COUNT(*) FROM "+self.current_ksp+".Test2StorageObj")[0]
        self.assertEqual(count, 1)
        self.assertEqual(so.name, 'caio')
        self.assertEqual(so.age, 1000)
        so.name = 'addio'
        so.age = 2000
        self.assertEqual(so.name, 'addio')
        self.assertEqual(so.age, 2000)

    def test_paranoid_setattr_nonpersistent(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        so = Test2StorageObj("test_paranoid_setattr_nonpersistent")
        so.name = 'my_name'
        self.assertEquals(so.name, 'my_name')

        def set_name_test():
            so.name = 1

        self.assertRaises(cassandra.InvalidRequest, set_name_test)
        so.age = 1
        self.assertEquals(so.age, 1)

        def set_age_test():
            so.age = 'my_name'

        self.assertRaises(cassandra.InvalidRequest, set_age_test)

    def test_paranoid_setattr_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        so = Test2StorageObj("t2")
        so.name = 'my_name'
        result = config.session.execute("SELECT name FROM "+self.current_ksp+".Test2StorageObj")
        for row in result:
            cass_name = row.name
        self.assertEquals(cass_name, 'my_name')

        def setNameTest():
            so.name = 1

        self.assertRaises(cassandra.InvalidRequest, setNameTest)
        so.age = 1
        result = config.session.execute("SELECT age FROM "+self.current_ksp+".Test2StorageObj")
        for row in result:
            cass_age = row.age
        self.assertEquals(cass_age, 1)

        def setAgeTest():
            so.age = 'my_name'

        self.assertRaises(cassandra.InvalidRequest, setAgeTest)

    def test_paranoid_setattr_float(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObjFloat")
        time.sleep(1)
        so = Test2StorageObjFloat("test_paranoid_setattr_float")
        so.age = 2.0

    def test_nestedso_notpersistent(self):

        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test3StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test4StorageObj")
        time.sleep(1)
        my_nested_so = Test3StorageObj()

        my_nested_so.myso.name = 'Link'
        self.assertEquals('Link', my_nested_so.myso.name)
        my_nested_so.myso.age = 10
        self.assertEquals(10, my_nested_so.myso.age)

        error = False
        try:
            config.session.execute('SELECT * FROM '+self.current_ksp+'.Test3StorageObj')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(True, error)

        my_nested_so.myso2.test[0] = 'position0'
        self.assertEquals('position0', my_nested_so.myso2.test[0])

        my_nested_so2 = Test4StorageObj()

        my_nested_so2.myotherso.name = 'Link'
        self.assertEquals('Link', my_nested_so2.myotherso.name)
        my_nested_so2.myotherso.age = 10
        self.assertEquals(10, my_nested_so2.myotherso.age)

        error = False
        try:
            config.session.execute('SELECT * FROM '+self.current_ksp+'.myso')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(True, error)

        my_nested_so3 = Test4bStorageObj('mynested')
        my_nested_subso = my_nested_so3.myotherso

        my_other_nested = getByID(my_nested_subso.storage_id)
        my_other_nested.name = 'bla'
        my_other_nested.age = 5
        error = False
        try:
            result = config.session.execute('SELECT * FROM '+self.current_ksp+'.Test2StorageObj')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(False, error)
        for row in result:
            query_res = row
        self.assertEquals(5, query_res.age)
        self.assertEquals('bla', query_res.name)

    def test_nestedso_persistent(self):

        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test3StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObj")
        time.sleep(1)
        my_nested_so = Test3StorageObj('test_nestedso_persistent')
        self.assertEquals(True, my_nested_so._is_persistent)
        self.assertEquals(True, my_nested_so.myso._is_persistent)
        self.assertEquals(True, my_nested_so.myso2._is_persistent)

        my_nested_so.myso.name = 'Link'
        my_nested_so.myso.age = 10
        error = False
        try:
            result = config.session.execute('SELECT * FROM '+self.current_ksp+'.test2storageobj')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(False, error)
        for row in result:
            query_res = row
        self.assertEquals(10, query_res.age)
        self.assertEquals('Link', query_res.name)

        my_nested_so.myso2.name = 'position0'
        self.assertEquals('position0', my_nested_so.myso2.name)

    def test_nestedso_topersistent(self):

        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test3StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        my_nested_so = Test3StorageObj()

        my_nested_so.myso.name = 'Link'
        self.assertEquals('Link', my_nested_so.myso.name)
        my_nested_so.myso.age = 10
        self.assertEquals(10, my_nested_so.myso.age)
        error = False
        try:
            result = config.session.execute('SELECT * FROM '+self.current_ksp+'.test2storageobj')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(True, error)

        my_nested_so.make_persistent('mynewso')

        error = False
        try:
            result = config.session.execute('SELECT * FROM '+self.current_ksp+'.test2storageobj')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(False, error)
        for row in result:
            query_res = row
        self.assertEquals(10, query_res.age)
        self.assertEquals('Link', query_res.name)

    def test_nestedso_sets_gets(self):

        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test3StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        my_nested_so = Test3StorageObj()

        my_nested_so.myso.name = 'Link'
        self.assertEquals('Link', my_nested_so.myso.name)
        my_nested_so.myso.age = 10
        self.assertEquals(10, my_nested_so.myso.age)
        my_nested_so.myso.weight = 70
        self.assertEquals(70, my_nested_so.myso.weight)
        #error = False
        #try:
        #    result = config.session.execute('SELECT * FROM '+self.current_ksp+'.test_nestedso_sets_gets_myso')
        #except cassandra.InvalidRequest:
        #    error = True
        #self.assertEquals(True, error)

        my_nested_so.make_persistent('test_nestedso_sets_gets_myso')

        error = False
        try:
            result = config.session.execute('SELECT * FROM '+self.current_ksp+'.Test2StorageObj')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(False, error)
        for row in result:
            query_res = row
        self.assertEquals(10, query_res.age)
        self.assertEquals('Link', query_res.name)
        error = False
        try:
            _ = query_res.weight
        except:
            error = True
        self.assertEquals(True, error)
        my_nested_so.myso.weight = 50
        self.assertEquals(50, my_nested_so.myso.weight)
        result = config.session.execute('SELECT * FROM '+self.current_ksp+'.Test2StorageObj')
        for row in result:
            query_res = row
        error = False
        try:
            _ = query_res.weight
        except:
            error = True
        self.assertEquals(True, error)

    def test_nestedso_sets_gets_complex(self):

        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test3StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        my_nested_so = Test3StorageObj()

        error = False
        try:
            _ = config.session.execute('SELECT * FROM '+self.current_ksp+'.TestStorageObj')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(True, error)

        my_nested_so.make_persistent('tnsgc')

        # We create the nested persistent objects only after they are accessed by the first time
        error = False
        try:
            _ = config.session.execute('SELECT * FROM '+self.current_ksp+'.TestStorageObj')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(True, error)

        for i in range(0, 100):
            my_nested_so.myso2.test[i] = 'position' + str(i)
        time.sleep(5)
        table_name = my_nested_so.myso2.test._table
        count, = config.session.execute("SELECT COUNT(*) FROM "+self.current_ksp+"." + table_name)[0]
        self.assertEquals(100, count)

    def test_nestedso_deletepersistent(self):

        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test3StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        my_nested_so = Test3StorageObj('tndp')

        self.assertEquals(True, my_nested_so._is_persistent)
        my_nested_so.myso.name = 'Link'
        self.assertEquals('Link', my_nested_so.myso.name)
        my_nested_so.myso.age = 10
        self.assertEquals(10, my_nested_so.myso.age)

        my_nested_so.delete_persistent()

        self.assertEquals(False, my_nested_so._is_persistent)
        entries = 0
        try:
            _ = config.session.execute('SELECT * FROM '+self.current_ksp+'.test2storageobj')
        except cassandra.InvalidRequest:
            entries += 1
        self.assertEquals(0, entries)

    def test_delete_persistent_obj(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        sobj = Test2StorageObj()
        sobj.name = "nom1"
        sobj.age = 42
        sobj.make_persistent("testpobj1")

        sobj2 = Test2StorageObj()
        sobj2.name = "nom2"
        sobj2.age = 666
        sobj2.make_persistent("testpobj2")

        sobj2.delete_persistent()
        del sobj2

        del sobj

        sobj3 = Test2StorageObj("testpobj1")
        self.assertEquals(sobj3.name, "nom1")

        sobj4 = Test2StorageObj("testpobj2")
        with self.assertRaises(AttributeError): # The object should be EMPTY
            name = sobj4.name

    def test_nestedso_dictofsos(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test5StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        my_nested_so = Test5StorageObj()
        my_nested_so.test2[0] = Test2StorageObj()
        my_nested_so.make_persistent('test_nestedso_dictofsos')
        self.assertEquals(True, my_nested_so._is_persistent)
        self.assertEquals(True, my_nested_so.test2._is_persistent)
        self.assertEquals(True, my_nested_so.test2[0]._is_persistent)

        my_nested_so.test2[0].name = 'Link'
        self.assertEquals('Link', my_nested_so.test2[0].name)
        my_nested_so.test2[0].age = 10
        self.assertEquals(10, my_nested_so.test2[0].age)

    def test_nestedso_dictofsos_noname(self):
        #'''
        #this test similar to test_nestedso_dictofsos with the difference that the StorageDict
        #used as an attribute in Test7StorageObj has the form <int,StorageObj> where no name has been given for the
        #StorageObj nor the Integer. In this case, a default name is used (key0,val0).
        #'''

        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test7StorageObj")
        time.sleep(1)
        my_nested_so = Test7StorageObj()
        my_nested_so.test2[0] = Test2StorageObj()
        my_nested_so.make_persistent('test_nestedso_dictofsos_noname')
        self.assertEquals(True, my_nested_so._is_persistent)
        self.assertEquals(True, my_nested_so.test2._is_persistent)
        self.assertEquals(True, my_nested_so.test2[0]._is_persistent)

        my_nested_so.test2[0].name = 'Link'
        self.assertEquals('Link', my_nested_so.test2[0].name)
        my_nested_so.test2[0].age = 10
        self.assertEquals(10, my_nested_so.test2[0].age)

    def test_nestedso_retrievedata(self):

        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test5StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        my_nested_so = Test5StorageObj('tnr')
        my_nested_so.test2[0] = Test2StorageObj('tnr_something')
        self.assertEquals(True, my_nested_so._is_persistent)
        self.assertEquals(True, my_nested_so.test2._is_persistent)
        self.assertEquals(True, my_nested_so.test2[0]._is_persistent)

        my_nested_so.test2[0].name = 'Link'
        self.assertEquals('Link', my_nested_so.test2[0].name)
        my_nested_so.test2[0].age = 10
        self.assertEquals(10, my_nested_so.test2[0].age)

        my_nested_so.sync()

        del my_nested_so

        my_nested_so2 = Test5StorageObj('tnr')

        self.assertEquals('Link', my_nested_so2.test2[0].name)
        self.assertEquals(10, my_nested_so2.test2[0].age)

    def test_numpy_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObjNumpy")
        time.sleep(1)
        my_so = TestStorageObjNumpy('tnp')

    def test_numpy_set(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObjNumpy")
        time.sleep(1)
        my_so = TestStorageObjNumpy()
        my_so.mynumpy = np.random.rand(3, 2)
        my_so.make_persistent('test_numpy_set')

    def test_numpy_get(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObjNumpy")
        time.sleep(1)
        my_so = TestStorageObjNumpy('test_numpy_get')
        mynumpy = np.random.rand(3, 2)
        my_so.mynumpy = mynumpy
        time.sleep(2)
        self.assertTrue(np.array_equal(mynumpy, my_so.mynumpy))

    def test_numpy_topersistent(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObjNumpy")
        time.sleep(1)
        my_so = TestStorageObjNumpy()
        my_so.mynumpy = np.random.rand(3, 2)
        my_so.make_persistent('test_numpy_topersistent')

    def test_numpydict_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObjNumpyDict")
        time.sleep(1)
        my_so = TestStorageObjNumpyDict('test_numpydict_persistent')

    def test_numpydict_set(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObjNumpyDict")
        time.sleep(1)
        my_so = TestStorageObjNumpyDict('test_numpydict_set')
        my_so.mynumpydict[0] = np.random.rand(3, 2)

    def test_numpydict_to_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObjNumpyDict")
        time.sleep(1)
        my_so = TestStorageObjNumpyDict()
        my_so.mynumpydict[0] = np.random.rand(3, 2)
        my_so.make_persistent('test_numpydict_to_persistent')

    def test_numpydict_get(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObjNumpyDict")
        time.sleep(1)
        my_so = TestStorageObjNumpyDict()
        mynumpydict = np.random.rand(3, 2)
        my_so.mynumpydict[0] = mynumpydict
        my_so.make_persistent('test_numpydict_get')
        time.sleep(2)
        self.assertTrue(np.allclose(mynumpydict, my_so.mynumpydict[0]))

    def test_numpy_operations(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObjNumpy")
        time.sleep(1)
        my_so = TestStorageObjNumpy()
        base_numpy = np.arange(2048)
        my_so.mynumpy = np.arange(2048)
        my_so.make_persistent('test_numpy_operations')
        time.sleep(2)
        self.assertTrue(np.array_equal(base_numpy, my_so.mynumpy))
        base_numpy += 1
        my_so.mynumpy += 1
        self.assertTrue(np.array_equal(base_numpy, my_so.mynumpy))
        self.assertEqual(np.average(base_numpy), np.average(my_so.mynumpy))
        self.assertEqual(np.mean(base_numpy), np.mean(my_so.mynumpy))

    def test_numpy_ops_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObjNumpy")
        time.sleep(1)
        my_so = TestStorageObjNumpy()
        base_numpy = np.arange(2048)
        my_so.mynumpy = np.arange(2048)
        my_so.make_persistent(self.current_ksp+'.test_numpy_ops_persistent')
        self.assertTrue(np.array_equal(base_numpy, my_so.mynumpy))
        base_numpy += 1
        my_so.mynumpy += 1
        self.assertTrue(np.array_equal(base_numpy, my_so.mynumpy))

        my_so.sync()
        reloaded_so = TestStorageObjNumpy(self.current_ksp+'.test_numpy_ops_persistent')
        self.assertTrue(np.allclose(reloaded_so.mynumpy, base_numpy))
        self.assertEqual(np.average(base_numpy), np.average(reloaded_so.mynumpy))
        self.assertEqual(np.mean(base_numpy), np.mean(reloaded_so.mynumpy))

    def test_numpy_reloading(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObjNumpy")
        time.sleep(1)
        sizea, sizeb = (1000, 1000)
        no = TestStorageObjNumpy(self.current_ksp +".test_numpy_reloading_%d_%d" % (sizea, sizeb))
        a = np.ones((sizea, sizeb))
        no.mynumpy = a
        del no
        import gc
        gc.collect()
        no = TestStorageObjNumpy(self.current_ksp +".test_numpy_reloading_%d_%d" % (sizea, sizeb))
        a = no.mynumpy
        self.assertEqual(np.shape(a), (sizea, sizeb))
        self.assertEqual(np.sum(a), sizea * sizeb)

    def test_numpy_reloading_internals(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObjNumpy")
        time.sleep(1)
        sizea, sizeb = (1000, 1000)
        no = TestStorageObjNumpy(self.current_ksp +".test_numpy_reloading_internals_%d_%d" % (sizea, sizeb))
        a = np.ones((sizea, sizeb))
        no.mynumpy = a
        initial_name_so = no._ksp + '.' + no._table
        initial_name_np = no.mynumpy._ksp + '.' + no.mynumpy._table
        del no
        import gc
        gc.collect()
        no = TestStorageObjNumpy(self.current_ksp +".test_numpy_reloading_internals_%d_%d" % (sizea, sizeb))
        a = no.mynumpy

        final_name_so = no._ksp + '.' + no._table
        final_name_np = no.mynumpy._ksp + '.' + no.mynumpy._table
        self.assertEqual(initial_name_so, final_name_so)
        self.assertEqual(initial_name_np, final_name_np)

    def test_storagedict_assign(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObj")
        time.sleep(1)
        so = TestStorageObj("t2_1")
        table_name = so.test._table
        so.test = {}
        self.assertNotEquals(table_name, so.test._table)
        so.test = {1: 'a', 2: 'b'}
        self.assertNotEquals(table_name, so.test._table)
        so.test = {3: 'c', 4: 'd'}
        self.assertNotEquals(table_name, so.test._table)

    def test_storageobj_coherence_basic(self):
        #'''
        #test that two StorageObjs pointing to the same table work correctly.
        #Changing data on one StorageObj is reflected on the other StorageObj.
        #'''
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        so = Test2StorageObj('test_storageobj_coherence_basic')
        so.name = 'Oliver'
        so.age = 21
        so2 = Test2StorageObj('test_storageobj_coherence_basic')
        self.assertEqual(so.name, so2.name)
        self.assertEqual(so.age, so2.age)
        so.name = 'Benji'
        so2 = Test2StorageObj('test_storageobj_coherence_basic')
        self.assertEqual(so.name, so2.name)
        self.assertEqual(so.age, so2.age)

    def test_storageobj_coherence_complex1(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test3StorageObj")
        time.sleep(1)
        so = Test3StorageObj('test_storageobj_coherence_complex1')
        myso_attr = Test2StorageObj()
        myso_attr.name = 'Oliver'
        myso_attr.age = 21
        so.myso = myso_attr  # creates my_app.test_myso_0, the original attribute pointed to test_myso
        self.assertEqual(myso_attr.name, so.myso.name)
        del myso_attr
        self.assertEqual(so.myso.age, 21)

    def test_storageobj_coherence_complex2(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestStorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test3StorageObj")
        time.sleep(1)
        so = Test3StorageObj('test_storageobj_coherence_complex2')
        myso_attr = Test2StorageObj()
        myso_attr.name = 'Oliver'
        myso_attr.age = 21
        so.myso = myso_attr  # creates my_app.test_myso_0, the original attribute pointed to test_myso
        # now my_attr is persistent too, because it has been asigned to a persistent object
        # Python behaviour, now the attribute points to the object, no copy made
        self.assertTrue(so.myso is myso_attr)
        # any change on the nested attribute should change the original and backwards
        attr_value = 123
        myso_attr.some_attribute = attr_value
        myso_attr.name = 'Benji'
        self.assertTrue(hasattr(so.myso, 'some_attribute'))
        self.assertEqual(so.myso.some_attribute, attr_value)
        self.assertEqual(so.myso.name, 'Benji')

        # now we unreference the top persistent object called so which was made persistent as 'test'
        del so

        # The object pointed by 'so.myso' should still exist because we still have one reference called 'myso_attr'

        self.assertTrue(myso_attr is not None)
        self.assertTrue(isinstance(myso_attr, Test2StorageObj))
        self.assertEqual(myso_attr.name, 'Benji')

    def test_get_attr_1(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestAttributes")
        time.sleep(1)
        storage_obj = TestAttributes()
        storage_obj.do_nothing_at_all()
        value = 123
        storage_obj.setvalue(value)
        returned = storage_obj.getvalue()
        self.assertEqual(value, returned)

    def test_get_attr_2(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestAttributes")
        time.sleep(1)
        storage_obj = TestAttributes()
        storage_obj.do_nothing_at_all()
        value = 123
        storage_obj.setvalue(value)
        storage_obj.make_persistent("test_get_attr_2")
        # check that the in memory attribute is kept
        returned = storage_obj.getvalue()
        self.assertEqual(value, returned)
        # check that the method added by inheritance is correctly called
        storage_obj.do_nothing_at_all()

        def method_nonexistent():
            storage_obj.i_dont_exist()

        # check that an attribute method which doesn't exist is detected
        self.assertRaises(AttributeError, method_nonexistent)

        # check for private methods too (starting with underscore)
        def method_nonexistent_pvt():
            storage_obj._pvt_dont_exist()

        self.assertRaises(AttributeError, method_nonexistent_pvt)


    def test_get_attr_3(self):
        # the same as test_get_attr_2 but the object is persistent since the beginning

        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestAttributes")
        time.sleep(1)
        storage_obj = TestAttributes("test_get_attr_3")
        storage_obj.do_nothing_at_all()
        value = 123
        storage_obj.setvalue(value)
        # check that the in memory attribute is kept
        returned = storage_obj.getvalue()
        self.assertEqual(value, returned)
        # check that the method added by inheritance is correctly called
        storage_obj.do_nothing_at_all()

        def method_nonexistent():
            storage_obj.i_dont_exist()

        # check that an attribute method which doesn't exist is detected
        self.assertRaises(AttributeError, method_nonexistent)

        # check for private methods too (starting with underscore)
        def method_nonexistent_pvt():
            storage_obj._pvt_dont_exist()

        self.assertRaises(AttributeError, method_nonexistent_pvt)

        storage_obj.key = 123
        returned = storage_obj.key
        self.assertEqual(storage_obj.key, returned)


    def test_recreation_init(self):
        #"""
        #New StorageObj
        #Persistent attributes
        #Made persistent on the constructor.
        #"""
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        sobj_name = self.current_ksp+".test_recreation_init"
        attr1 = 'Test1'
        attr2 = 23
        storage_obj = Test2StorageObj(sobj_name)
        storage_obj.name = attr1
        storage_obj.age = attr2
        uuid_sobj = storage_obj.storage_id

        storage_obj = None
        result_set = iter(config.session.execute("SELECT * FROM hecuba.istorage WHERE storage_id={}".format(uuid_sobj)))

        try:
            result = result_set.next()
        except StopIteration as ex:
            self.fail("StorageObj istorage data was not saved")

        self.assertEqual(result.name, sobj_name)

        storage_obj = Test2StorageObj(sobj_name)

        self.assertEqual(storage_obj.name, attr1)
        self.assertEqual(storage_obj.age, attr2)

    def test_recreation_init2(self):
        #"""
        #New StorageObj
        #Has persistent and volatile attributes
        #Made persistent on the constructor.
        #"""
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        sobj_name = self.current_ksp +".test_recreation_init2"
        attr1 = 'Test1'
        attr2 = 23
        storage_obj = Test2StorageObj(sobj_name)
        storage_obj.name = attr1
        storage_obj.nonpersistent = attr2
        uuid_sobj = storage_obj.storage_id

        storage_obj = None
        result_set = iter(config.session.execute("SELECT * FROM hecuba.istorage WHERE storage_id={}".format(uuid_sobj)))

        try:
            result = result_set.next()
        except StopIteration as ex:
            self.fail("StorageObj istorage data was not saved")

        self.assertEqual(result.name, sobj_name)

        storage_obj = Test2StorageObj(sobj_name)

        self.assertEqual(storage_obj.name, attr1)

        with self.assertRaises(AttributeError):
            attr = storage_obj.age

        with self.assertRaises(AttributeError):
            attr = storage_obj.nonpersistent

    def test_recreation_make_pers(self):
        #"""
        #New StorageObj
        #Persistent attributes
        #Made persistent with make_persistent.
        #"""
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        sobj_name = self.current_ksp +".test_recreation_make_pers"
        attr1 = 'Test1'
        attr2 = 23
        storage_obj = Test2StorageObj()
        storage_obj.make_persistent(sobj_name)
        uuid_sobj = storage_obj.storage_id

        storage_obj = None
        result_set = iter(config.session.execute("SELECT * FROM hecuba.istorage WHERE storage_id={}".format(uuid_sobj)))

        try:
            result = result_set.next()
        except StopIteration as ex:
            self.fail("StorageObj istorage data was not saved")

        self.assertEqual(result.name, sobj_name)

        storage_obj = Test2StorageObj()

        storage_obj.name = attr1
        storage_obj.volatile = attr2

        storage_obj.make_persistent(sobj_name)

        self.assertEqual(storage_obj.name, attr1)
        self.assertEqual(storage_obj.volatile, attr2)

        with self.assertRaises(AttributeError):
            attr = storage_obj.age

    def test_recreation_make_pers2(self):
        #"""
        #New StorageObj
        #Persistent attributes
        #Made persistent with make_persistent.
        #"""
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        time.sleep(1)
        sobj_name = self.current_ksp+".test_recreation_make_pers2"
        attr1 = 'Test1'
        attr2 = 23
        storage_obj = Test2StorageObj()
        storage_obj.name = attr1
        storage_obj.volatile = 'Ofcourse'
        storage_obj.make_persistent(sobj_name)
        uuid_sobj = storage_obj.storage_id

        storage_obj = None
        result_set = iter(config.session.execute("SELECT * FROM hecuba.istorage WHERE storage_id={}".format(uuid_sobj)))

        try:
            result = result_set.next()
        except StopIteration as ex:
            self.fail("StorageObj istorage data was not saved")

        self.assertEqual(result.name, sobj_name)

        storage_obj = Test2StorageObj()
        storage_obj.age = attr2
        storage_obj.make_persistent(sobj_name)

        self.assertEqual(storage_obj.name, attr1)
        self.assertEqual(storage_obj.age, attr2)

        with self.assertRaises(AttributeError):
            attr = storage_obj.volatile

    def test_nested_recreation(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test2StorageObj")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test4StorageObj")
        time.sleep(1)
        sobj_name = self.current_ksp +".test_nested_recreation"

        storage_obj = Test2StorageObj()
        name_attr = 'Test1'
        age_attr = 23
        storage_obj.name = name_attr
        storage_obj.age = age_attr

        external_sobj = Test4StorageObj(sobj_name)
        external_sobj.myotherso = storage_obj

        uuid_sobj_internal = storage_obj.storage_id
        uuid_sobj_external = external_sobj.storage_id

        internal_name = external_sobj.myotherso._get_name()
        storage_obj = None
        external_sobj = None

        # Check that they have been correctly stored into hecuba.istorage

        result_set = iter(
            config.session.execute("SELECT * FROM hecuba.istorage WHERE storage_id={}".format(uuid_sobj_external)))

        try:
            result = result_set.next()
        except StopIteration as exc:
            self.fail("StorageObj istorage data was not saved")

        self.assertEqual(result.name, sobj_name)

        result_set = iter(
            config.session.execute("SELECT * FROM hecuba.istorage WHERE storage_id={}".format(uuid_sobj_internal)))

        try:
            result = result_set.next()
        except StopIteration as exc:
            self.fail("StorageObj istorage data was not saved")

        self.assertEqual(result.name, internal_name)

        # They are both present in hecuba.istorage

        result_set = iter(
            config.session.execute(
                "SELECT * FROM {} WHERE storage_id={}".format(self.current_ksp+".Test4StorageObj", uuid_sobj_external)))

        try:
            result = result_set.next()
        except StopIteration as exc:
            self.fail("StorageObj istorage data was not saved")

        self.assertEqual(result.myotherso, uuid_sobj_internal)
        # They have been saved with the expected istorage ids

        external_sobj = Test4StorageObj(sobj_name)
        # Check internal configuration is correct
        self.assertEqual(external_sobj.storage_id, uuid_sobj_external)
        self.assertEqual(external_sobj.myotherso.storage_id, uuid_sobj_internal)
        self.assertEqual(external_sobj._get_name(), sobj_name)
        self.assertEqual(external_sobj.myotherso._get_name(), internal_name)

        # Check data is correct
        self.assertEqual(external_sobj.myotherso.name, name_attr)
        self.assertEqual(external_sobj.myotherso.age, age_attr)

    def test_single_table(self):
        config.session.execute("DROP TABLE IF EXISTS " + self.current_ksp+".Test2StorageObj")
        time.sleep(1)
        my_obj1 = Test2StorageObj(self.current_ksp+".test_single_tablemy_obj1")
        my_obj2 = Test2StorageObj(self.current_ksp+".test_single_tablemy_obj2")
        my_obj1.name, my_obj2.name = "Adrian", "Adri"
        my_obj1.age, my_obj2.age = 21, 23
        self.assertEqual(my_obj1._ksp, my_obj2._ksp)
        self.assertEqual(my_obj1._table, my_obj2._table)

        res = config.session.execute("SELECT * FROM "+self.current_ksp+".Test2StorageObj WHERE storage_id = %s" % my_obj1.storage_id)
        res2 = config.session.execute(
            "SELECT * FROM "+self.current_ksp+".Test2StorageObj WHERE storage_id = %s" % my_obj2.storage_id)
        self.assertEqual(res.one().name, "Adrian")
        self.assertEqual(res2.one().name, "Adri")
        self.assertEqual(res.one().age, 21)
        self.assertEqual(res2.one().age, 23)

    def test_dict_single_table(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".Test5StorageObj")
        time.sleep(1)
        my_dict = Test5StorageObj(self.current_ksp+".my_dict4")
        for i in range(0, 20):
            aux = Test2StorageObj(self.current_ksp+".test_dict_single_table" + str(i))
            aux.name, aux.age = "RandomName" + str(i), 18 + i
            my_dict.test2[i] = aux

        for i in range(0, 20):
            self.assertEqual(my_dict.test2[i]._ksp, self.current_ksp)
            self.assertEqual(my_dict.test2[i]._table, "Test2StorageObj".lower())
            res = config.session.execute(
                    "SELECT * FROM "+self.current_ksp+".Test2StorageObj WHERE storage_id = %s" % my_dict.test2[i].storage_id)
            self.assertEqual(res.one().name, "RandomName" + str(i))
            self.assertEqual(res.one().age, 18 + i)

    def test_time(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestTime")
        time.sleep(1)
        d = TestTime(self.current_ksp+".test_time")

        mytime =datetime.time(hour=11, minute=43, second=2, microsecond=90)
        d.attr = mytime
        del d
        mynew_d = TestTime(self.current_ksp+".test_time")
        self.assertEqual(mynew_d.attr, mytime)

    def test_date(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestDate")
        time.sleep(1)
        d = TestDate(self.current_ksp +".test_date")

        mydate = datetime.date(year=1992, month=7, day=25)
        d.attr = mydate
        del d
        mynew_d = TestDate(self.current_ksp+".test_date")

        self.assertEqual(mynew_d.attr, mydate)

    def test_datetime(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".TestDateTime")
        time.sleep(1)
        d = TestDateTime(self.current_ksp+".test_datetime")
        dtime = datetime.datetime(year=1940, month=10, day=16,
                         hour=23, minute=59, second=59)
        d.attr = dtime
        del d
        mynew_d = TestDateTime(self.current_ksp+".test_datetime")
        self.assertEqual(mynew_d.attr, dtime)

    def test_storageobjdict_unnamed(self):
        d = TestStorageObjDict("test_sobjdict_unnamed")
        d.MyAttribute_2[1]="hola"
        d.MyAttribute_3[[42,"hola"]]=666
        d.sync()
        d = TestStorageObjDict("test_sobjdict_unnamed")
        self.assertEqual(d.MyAttribute_2[1], "hola")
        self.assertEqual(d.MyAttribute_3[[42,"hola"]], 666)

    def test_so_schemas(self):
        class tsoschemasModel(StorageObj):
            '''
            @ClassField uno str
            @ClassField dos int
            '''
        m = tsoschemasModel()
        m.uno="hola"
        m.dos=42
        m.make_persistent("test_so_schemas")

        class tsoschemasModel2(StorageObj):
            '''
            @ClassField uno int
            @ClassField dos str
            '''
        m = tsoschemasModel2()
        m.uno=42
        m.dos="hola"
        with self.assertRaises(RuntimeError):
            m.make_persistent("test_so_schemas") # Same name, but different schema. SHOULD FAIL

        #class tsoschemasModel(StorageObj):
        #    '''
        #    @ClassField uno int
        #    @ClassField dos str
        #    '''
        #m = tsoschemasModel()
        #m.uno=42
        #m.dos="hola"
        #with self.assertRaises(RuntimeError):
        #    m.make_persistent("test_so_schemas2") # Same classname, but different schema. SHOULD FAIL

        class tsoschemasModel3(StorageObj):
            '''
            @ClassField uno int
            @ClassField dos str
            @ClassField tres dict<<int>, str>
            '''
        m = tsoschemasModel3()
        m.uno=42
        m.dos="hola"
        m.tres[666]="adios"
        m.make_persistent("test_so_schemas3")
        m.sync()
        m = tsoschemasModel3("test_so_schemas3")
        self.assertEqual(m.uno, 42)
        self.assertEqual(m.dos, "hola")
        self.assertEqual(m.tres[666], "adios")

        with self.assertRaises(RuntimeError):
            m = tsoschemasModel("test_so_schemas3") # Same name, but different schema

if __name__ == '__main__':
    unittest.main()
