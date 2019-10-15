import time as user_time
import unittest
import uuid
from datetime import datetime

import cassandra
import numpy as np
from hecuba import config
from hecuba.IStorage import _discrete_token_ranges
from hecuba.storageobj import StorageObj
from storage.api import getByID

from ..app.words import Words


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

class TestTimestamp(StorageObj):
    '''
    @ClassField attr timestamp
    '''

class StorageObjTest(unittest.TestCase):
    def test_build_remotely(self):
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + ".tt1")
        obj = TestStorageObj(config.execution_name + ".tt1")
        r = {"built_remotely": False, "storage_id": uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.tt1'),
             "ksp": config.execution_name,
             "class_name": str(TestStorageObj.__module__) + "." + TestStorageObj.__name__, "name": 'tt1',
             "columns": [('val1', 'str')], "entry_point": 'localhost', "primary_keys": [('pk1', 'int')],
             "istorage_props": {},
             "tokens": _discrete_token_ranges([token.value for token in config.cluster.metadata.token_map.ring])}

        nopars = StorageObj.build_remotely(r)
        self.assertEqual('tt1', nopars._table)
        self.assertEqual(config.execution_name, nopars._ksp)
        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.tt1'), nopars._storage_id)
        name, tkns = \
            config.session.execute("SELECT name, tokens FROM hecuba.istorage WHERE storage_id = %s",
                                   [nopars._storage_id])[
                0]

        self.assertEqual(name, config.execution_name + '.tt1')
        self.assertEqual(tkns, r['tokens'])

    def test_init_create_pdict(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tt1")
        config.session.execute("DROP TABLE IF EXISTS my_app.tt1_instances")
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + '.tt1')

        r = {"built_remotely": False, "storage_id": uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.tt1'),
             "ksp": config.execution_name,
             "class_name": str(TestStorageObj.__module__) + "." + TestStorageObj.__name__, "name": 'tt1',
             "columns": [('val1', 'str')], "entry_point": 'localhost', "primary_keys": [('pk1', 'int')],
             "istorage_props": {},
             "tokens": _discrete_token_ranges([token.value for token in config.cluster.metadata.token_map.ring])}

        nopars = StorageObj.build_remotely(r)
        self.assertEqual(nopars._built_remotely, False)
        self.assertEqual('tt1', nopars._table)
        self.assertEqual(config.execution_name, nopars._ksp)
        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.tt1'), nopars._storage_id)
        name, tkns = \
            config.session.execute("SELECT name,tokens FROM hecuba.istorage WHERE storage_id = %s",
                                   [nopars._storage_id])[0]
        self.assertEqual(name, config.execution_name + '.' + r['name'])
        self.assertEqual(tkns, r['tokens'])

        tkns = _discrete_token_ranges(
            [8508619251581300691, 8514581128764531689, 8577968535836399533, 8596162846302799189,
             8603491526474728284, 8628291680139169981, 8687301163739303017, 9111581078517061776])
        config.session.execute("DROP TABLE IF EXISTS " + config.execution_name + '.tt2')
        nopars = Result(name='tt2',
                        tokens=tkns)
        self.assertEqual('tt2', nopars._table)
        self.assertEqual(config.execution_name, nopars._ksp)
        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.tt2'), nopars._storage_id)
        self.assertEqual(True, nopars._is_persistent)
        self.assertTrue(hasattr(nopars, 'instances'))
        name, read_tkns = config.session.execute("SELECT name,tokens FROM hecuba.istorage WHERE storage_id = %s",
                                                 [nopars._storage_id])[0]

        self.assertEqual(name, config.execution_name + '.tt2')
        self.assertEqual(tkns, read_tkns)

    def test_mixed_class(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba_test.bla")
        myObj = mixObj()

        myObj.make_persistent("hecuba_test.bla")

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
                                   "FROM hecuba_test.bla WHERE storage_id =" + str(myObj._storage_id))[0]

        self.assertEquals(floatfield, myObj.floatfield)
        self.assertEquals(intField, myObj.intField)
        self.assertEquals(strField, myObj.strField)
        self.assertEquals(intlistField, myObj.intlistField)
        self.assertEquals(floatlistField, myObj.floatlistField)
        self.assertEquals(strlistField, myObj.strlistField)
        self.assertEquals(inttupleField, myObj.inttupleField)

    def test_init_empty(self):
        config.session.execute("DROP TABLE IF EXISTS ksp1.ttta")
        nopars = TestStorageObj('ksp1.ttta')
        self.assertEqual('ttta', nopars._table)
        self.assertEqual('ksp1', nopars._ksp)

        res = config.session.execute(
            'SELECT storage_id, class_name, name, tokens, istorage_props FROM hecuba.istorage WHERE storage_id = %s',
            [nopars._storage_id])[0]

        storage_id, storageobj_classname, name, tokens, istorage_props = res
        self.assertEqual(storage_id, nopars._storage_id)
        self.assertEqual(storageobj_classname, TestStorageObj.__module__ + "." + TestStorageObj.__name__)
        self.assertEqual(name, 'ksp1.ttta')

        rebuild = StorageObj.build_remotely(res._asdict())
        self.assertEqual(rebuild._built_remotely, True)
        self.assertEqual('ttta', rebuild._table)
        self.assertEqual('ksp1', rebuild._ksp)
        self.assertEqual(storage_id, rebuild._storage_id)

        self.assertEqual(nopars._is_persistent, rebuild._is_persistent)
        # self.assertEqual(vars(nopars), vars(rebuild))

    def test_make_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba_test.wordsso")
        config.session.execute("DROP TABLE IF EXISTS hecuba_test.nonames")
        config.session.execute("DROP TABLE IF EXISTS hecuba_test.words")
        config.session.execute("DROP TABLE IF EXISTS hecuba_test.wordsso_words")
        config.session.execute("DROP TABLE IF EXISTS hecuba_test.nonames_test3")
        nopars = Words()
        self.assertFalse(nopars._is_persistent)
        nopars.ciao = 1
        nopars.ciao2 = "1"
        nopars.ciao3 = [1, 2, 3]
        nopars.ciao4 = (1, 2, 3)
        for i in range(10):
            nopars.words[i] = ['ciao' + str(i)]

        count, = config.session.execute(
            "SELECT count(*) FROM system_schema.tables WHERE keyspace_name = 'hecuba_test' and table_name = 'words'")[0]
        self.assertEqual(0, count)

        nopars.make_persistent("hecuba_test.wordsso")
        del nopars
        count, = config.session.execute('SELECT count(*) FROM hecuba_test.wordsso_words')[0]
        self.assertEqual(10, count)

        nopars2 = Test6StorageObj("hecuba_test.nonames")
        nopars2.test3[0] = ['1', '2']
        user_time.sleep(2)
        result = config.session.execute("SELECT val0, val1 FROM hecuba_test.nonames_test3 WHERE key0 = 0")

        rval0 = None
        rval1 = None
        for row in result:
            rval0 = row.val0
            rval1 = row.val1

        self.assertEqual('1', rval0)
        self.assertEqual('2', rval1)

    def test_empty_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.wordsso_words")
        config.session.execute("DROP TABLE IF EXISTS my_app.wordsso")
        so = Words()
        so.make_persistent("wordsso")
        so.ciao = "an attribute"
        so.another = 123
        config.batch_size = 1
        config.cache_activated = False
        for i in range(10):
            so.words[i] = str.join(',', map(lambda a: "ciao", range(i)))

        del so

        count, = config.session.execute('SELECT count(*) FROM my_app.wordsso_words')[0]
        self.assertEqual(10, count)
        so = Words()
        so.make_persistent("wordsso")
        so.delete_persistent()

        count, = config.session.execute('SELECT count(*) FROM my_app.wordsso_words')[0]
        self.assertEqual(0, count)

    def test_simple_stores_after_make_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.t2")
        so = Test2StorageObj()
        so.name = 'caio'
        so.age = 1000
        so.make_persistent("t2")
        count, = config.session.execute("SELECT COUNT(*) FROM my_app.t2")[0]
        self.assertEqual(count, 1)
        self.assertEqual(so.name, 'caio')
        self.assertEqual(so.age, 1000)

    def test_simple_attributes(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.t2")
        so = Test2StorageObj()
        so.make_persistent("t2")
        so.name = 'caio'
        so.age = 1000
        count, = config.session.execute("SELECT COUNT(*) FROM my_app.t2")[0]
        self.assertEqual(count, 1)
        self.assertEqual(so.name, 'caio')
        self.assertEqual(so.age, 1000)

    def test_modify_simple_attributes(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.t2")
        so = Test2StorageObj()
        so.make_persistent("t2")
        so.name = 'caio'
        so.age = 1000
        count, = config.session.execute("SELECT COUNT(*) FROM my_app.t2")[0]
        self.assertEqual(count, 1)
        self.assertEqual(so.name, 'caio')
        self.assertEqual(so.age, 1000)
        so.name = 'addio'
        so.age = 2000
        self.assertEqual(so.name, 'addio')
        self.assertEqual(so.age, 2000)

    def test_delattr_nonpersistent(self):
        so = Test2StorageObj()
        so.name = 'caio'
        del so.name

        def del_attr():
            my_val = so.name

        self.assertRaises(AttributeError, del_attr)

    def test_delattr_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.t3")
        so = Test2StorageObj("t3")
        so.name = 'caio'
        del so.name

        def del_attr1():
            my_val = so.name

        self.assertRaises(AttributeError, del_attr1)

        def del_attr2():
            my_val = so.random_val

        self.assertRaises(AttributeError, del_attr1)

    def test_delattr_persistent_nested(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.t4")
        so = Test3StorageObj("t4")
        nestedSo = Test2StorageObj()
        nestedSo.name = 'caio'
        so.myso = nestedSo
        # Make sure the inner object has been made persistent
        self.assertTrue(nestedSo._is_persistent)
        # Delete the attribute
        del so.myso

        def del_attr1():
            my_val = so.myso

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
        config.session.execute("DROP TABLE IF EXISTS my_app.t2")
        so = Test2StorageObj()
        so.name = 'caio'
        so.age = 1000
        so.make_persistent("t2")
        count, = config.session.execute("SELECT COUNT(*) FROM my_app.t2")[0]
        self.assertEqual(count, 1)
        self.assertEqual(so.name, 'caio')
        self.assertEqual(so.age, 1000)
        so.name = 'addio'
        so.age = 2000
        self.assertEqual(so.name, 'addio')
        self.assertEqual(so.age, 2000)

    def test_paranoid_setattr_nonpersistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.t2")
        so = Test2StorageObj("myobj")
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
        config.session.execute("DROP TABLE IF EXISTS my_app.t2")
        so = Test2StorageObj("t2")
        so.name = 'my_name'
        result = config.session.execute("SELECT name FROM my_app.t2")
        for row in result:
            cass_name = row.name
        self.assertEquals(cass_name, 'my_name')

        def setNameTest():
            so.name = 1

        self.assertRaises(cassandra.InvalidRequest, setNameTest)
        so.age = 1
        result = config.session.execute("SELECT age FROM my_app.t2")
        for row in result:
            cass_age = row.age
        self.assertEquals(cass_age, 1)

        def setAgeTest():
            so.age = 'my_name'

        self.assertRaises(cassandra.InvalidRequest, setAgeTest)

    def test_paranoid_setattr_float(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.t2")
        so = Test2StorageObjFloat("t2")
        so.age = 2.0

    def test_nestedso_notpersistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso")
        config.session.execute("DROP TABLE IF EXISTS my_app.myso")

        my_nested_so = Test3StorageObj()

        my_nested_so.myso.name = 'Link'
        self.assertEquals('Link', my_nested_so.myso.name)
        my_nested_so.myso.age = 10
        self.assertEquals(10, my_nested_so.myso.age)

        error = False
        try:
            config.session.execute('SELECT * FROM my_app.myso')
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
            config.session.execute('SELECT * FROM my_app.myso')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(True, error)

        my_nested_so3 = Test4bStorageObj('mynested')
        my_nested_subso = my_nested_so3.myotherso

        my_other_nested = getByID(my_nested_subso.getID())
        my_other_nested.name = 'bla'
        my_other_nested.age = 5
        error = False
        try:
            result = config.session.execute('SELECT * FROM my_app.mynested_myotherso')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(False, error)
        for row in result:
            query_res = row
        self.assertEquals(5, query_res.age)
        self.assertEquals('bla', query_res.name)

    def test_nestedso_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso")
        config.session.execute("DROP TABLE IF EXISTS my_app.myso")

        my_nested_so = Test3StorageObj('mynewso')
        self.assertEquals(True, my_nested_so._is_persistent)
        self.assertEquals(True, my_nested_so.myso._is_persistent)
        self.assertEquals(True, my_nested_so.myso2._is_persistent)

        my_nested_so.myso.name = 'Link'
        my_nested_so.myso.age = 10
        error = False
        try:
            result = config.session.execute('SELECT * FROM my_app.mynewso_myso')
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
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso")
        config.session.execute("DROP TABLE IF EXISTS my_app.myso")

        my_nested_so = Test3StorageObj()

        my_nested_so.myso.name = 'Link'
        self.assertEquals('Link', my_nested_so.myso.name)
        my_nested_so.myso.age = 10
        self.assertEquals(10, my_nested_so.myso.age)
        error = False
        try:
            result = config.session.execute('SELECT * FROM my_app.myso')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(True, error)

        my_nested_so.make_persistent('mynewso')

        error = False
        try:
            result = config.session.execute('SELECT * FROM my_app.mynewso_myso')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(False, error)
        for row in result:
            query_res = row
        self.assertEquals(10, query_res.age)
        self.assertEquals('Link', query_res.name)

    def test_nestedso_sets_gets(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.myso")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_myso")

        my_nested_so = Test3StorageObj()

        my_nested_so.myso.name = 'Link'
        self.assertEquals('Link', my_nested_so.myso.name)
        my_nested_so.myso.age = 10
        self.assertEquals(10, my_nested_so.myso.age)
        my_nested_so.myso.weight = 70
        self.assertEquals(70, my_nested_so.myso.weight)
        error = False
        try:
            result = config.session.execute('SELECT * FROM my_app.mynewso_myso')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(True, error)

        my_nested_so.make_persistent('mynewso')

        error = False
        try:
            result = config.session.execute('SELECT * FROM my_app.mynewso_myso')
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
        result = config.session.execute('SELECT * FROM my_app.mynewso_myso')
        for row in result:
            query_res = row
        error = False
        try:
            _ = query_res.weight
        except:
            error = True
        self.assertEquals(True, error)

    def test_nestedso_sets_gets_complex(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tnsgc_myso2")
        config.session.execute("DROP TABLE IF EXISTS my_app.tnsgc_myso2_test")

        my_nested_so = Test3StorageObj()

        error = False
        try:
            _ = config.session.execute('SELECT * FROM my_app.tnsgc_myso2')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(True, error)

        my_nested_so.make_persistent('tnsgc')

        error = False
        try:
            _ = config.session.execute('SELECT * FROM my_app.tnsgc_myso2')
        except cassandra.InvalidRequest:
            error = True
        self.assertEquals(False, error)

        for i in range(0, 100):
            my_nested_so.myso2.test[i] = 'position' + str(i)
        user_time.sleep(5)
        count, = config.session.execute("SELECT COUNT(*) FROM my_app.tnsgc_myso2_test")[0]
        self.assertEquals(100, count)

    def test_nestedso_deletepersistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tndp")
        config.session.execute("DROP TABLE IF EXISTS my_app.tndp_myso")
        config.session.execute("DROP TABLE IF EXISTS my_app.tndp_myso2")

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
            _ = config.session.execute('SELECT * FROM my_app.tndp_myso')
        except cassandra.InvalidRequest:
            entries += 1
        self.assertEquals(0, entries)

    def test_nestedso_dictofsos(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.topstorageobj")
        config.session.execute("DROP TABLE IF EXISTS my_app.topstorageobj_test2")
        my_nested_so = Test5StorageObj()
        my_nested_so.test2[0] = Test2StorageObj()
        my_nested_so.make_persistent('topstorageobj')
        self.assertEquals(True, my_nested_so._is_persistent)
        self.assertEquals(True, my_nested_so.test2._is_persistent)
        self.assertEquals(True, my_nested_so.test2[0]._is_persistent)

        my_nested_so.test2[0].name = 'Link'
        self.assertEquals('Link', my_nested_so.test2[0].name)
        my_nested_so.test2[0].age = 10
        self.assertEquals(10, my_nested_so.test2[0].age)

    def test_nestedso_dictofsos_noname(self):
        '''
        this test similar to test_nestedso_dictofsos with the difference that the StorageDict
        used as an attribute in Test7StorageObj has the form <int,StorageObj> where no name has been given for the
        StorageObj nor the Integer. In this case, a default name is used (key0,val0).
        '''
        config.session.execute("DROP TABLE IF EXISTS my_app.topstorageobj2")
        config.session.execute("DROP TABLE IF EXISTS my_app.topstorageobj2_test2")
        config.session.execute("DROP TABLE IF EXISTS my_app.topstorageobj2_test2_val0")

        my_nested_so = Test7StorageObj()
        my_nested_so.test2[0] = Test2StorageObj()
        my_nested_so.make_persistent('topstorageobj2')
        self.assertEquals(True, my_nested_so._is_persistent)
        self.assertEquals(True, my_nested_so.test2._is_persistent)
        self.assertEquals(True, my_nested_so.test2[0]._is_persistent)

        my_nested_so.test2[0].name = 'Link'
        self.assertEquals('Link', my_nested_so.test2[0].name)
        my_nested_so.test2[0].age = 10
        self.assertEquals(10, my_nested_so.test2[0].age)

    def test_nestedso_retrievedata(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tnr")
        config.session.execute("DROP TABLE IF EXISTS my_app.tnr_test2")

        my_nested_so = Test5StorageObj('tnr')
        my_nested_so.test2[0] = Test2StorageObj('something')
        self.assertEquals(True, my_nested_so._is_persistent)
        self.assertEquals(True, my_nested_so.test2._is_persistent)
        self.assertEquals(True, my_nested_so.test2[0]._is_persistent)

        my_nested_so.test2[0].name = 'Link'
        self.assertEquals('Link', my_nested_so.test2[0].name)
        my_nested_so.test2[0].age = 10
        self.assertEquals(10, my_nested_so.test2[0].age)

        del my_nested_so

        my_nested_so2 = Test5StorageObj('tnr')

        self.assertEquals('Link', my_nested_so2.test2[0].name)
        self.assertEquals(10, my_nested_so2.test2[0].age)

    def test_numpy_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tnp")
        config.session.execute("DROP TABLE IF EXISTS my_app.tnp_mynumpy")
        config.session.execute("DROP TABLE IF EXISTS my_app.tnp_mynumpy_numpies")
        my_so = TestStorageObjNumpy('tnp')

    def test_numpy_set(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpy")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpy_numpies")
        my_so = TestStorageObjNumpy()
        my_so.mynumpy = np.random.rand(3, 2)
        my_so.make_persistent('mynewso')

    def test_numpy_get(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpy")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpy_numpies")
        my_so = TestStorageObjNumpy('mynewso')
        mynumpy = np.random.rand(3, 2)
        my_so.mynumpy = mynumpy
        user_time.sleep(2)
        self.assertTrue(np.array_equal(mynumpy, my_so.mynumpy))

    def test_numpy_topersistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpy")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpy_numpies")
        my_so = TestStorageObjNumpy()
        my_so.mynumpy = np.random.rand(3, 2)
        my_so.make_persistent('mynewso')

    def test_numpydict_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpydict")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpydict_numpies")
        my_so = TestStorageObjNumpyDict('mynewso')

    def test_numpydict_set(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpydict")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpydict_numpies")
        my_so = TestStorageObjNumpyDict('mynewso')
        my_so.mynumpydict[0] = np.random.rand(3, 2)

    def test_numpydict_to_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpydict")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpydict_numpies")
        my_so = TestStorageObjNumpyDict()
        my_so.mynumpydict[0] = np.random.rand(3, 2)
        my_so.make_persistent('mynewso')

    def test_numpydict_get(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpydict")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpydict_numpies")
        my_so = TestStorageObjNumpyDict()
        mynumpydict = np.random.rand(3, 2)
        my_so.mynumpydict[0] = mynumpydict
        my_so.make_persistent('mynewso')
        user_time.sleep(2)
        self.assertTrue(np.array_equal(mynumpydict, my_so.mynumpydict[0]))

    def test_numpy_operations(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpy")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpy_numpies")
        my_so = TestStorageObjNumpy()
        base_numpy = np.arange(2048)
        my_so.mynumpy = np.arange(2048)
        my_so.make_persistent('mynewso')
        user_time.sleep(2)
        self.assertTrue(np.array_equal(base_numpy, my_so.mynumpy))
        base_numpy += 1
        my_so.mynumpy += 1
        self.assertTrue(np.array_equal(base_numpy, my_so.mynumpy))
        self.assertEqual(np.average(base_numpy), np.average(my_so.mynumpy))
        self.assertEqual(np.mean(base_numpy), np.mean(my_so.mynumpy))

    def test_numpy_ops_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpy")
        config.session.execute("DROP TABLE IF EXISTS my_app.mynewso_mynumpy_numpies")
        my_so = TestStorageObjNumpy()
        base_numpy = np.arange(2048)
        my_so.mynumpy = np.arange(2048)
        my_so.make_persistent('mynewso')
        user_time.sleep(2)
        self.assertTrue(np.array_equal(base_numpy, my_so.mynumpy))
        base_numpy += 1
        my_so.mynumpy += 1
        self.assertTrue(np.array_equal(base_numpy, my_so.mynumpy))

        reloaded_so = TestStorageObjNumpy('mynewso')
        self.assertTrue(np.array_equal(base_numpy, reloaded_so.mynumpy))
        self.assertEqual(np.average(base_numpy), np.average(reloaded_so.mynumpy))
        self.assertEqual(np.mean(base_numpy), np.mean(reloaded_so.mynumpy))

    def test_numpy_reloading(self):
        sizea, sizeb = (1000, 1000)
        no = TestStorageObjNumpy("my_app.numpy_test_%d_%d" % (sizea, sizeb))
        a = np.ones((sizea, sizeb))
        no.mynumpy = a
        del no
        no = TestStorageObjNumpy("my_app.numpy_test_%d_%d" % (sizea, sizeb))
        a = no.mynumpy
        self.assertEqual(np.shape(a), (sizea, sizeb))
        self.assertEqual(np.sum(a), sizea * sizeb)

    def test_numpy_reloading_internals(self):
        sizea, sizeb = (1000, 1000)
        no = TestStorageObjNumpy("my_app.numpy_test_%d_%d" % (sizea, sizeb))
        a = np.ones((sizea, sizeb))
        no.mynumpy = a
        initial_name_so = no._ksp + '.' + no._table
        initial_name_np = no.mynumpy._ksp + '.' + no.mynumpy._table
        del no
        no = TestStorageObjNumpy("my_app.numpy_test_%d_%d" % (sizea, sizeb))
        a = no.mynumpy

        final_name_so = no._ksp + '.' + no._table
        final_name_np = no.mynumpy._ksp + '.' + no.mynumpy._table
        self.assertEqual(initial_name_so, final_name_so)
        self.assertEqual(initial_name_np, final_name_np)

    def test_storagedict_assign(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.t2")
        config.session.execute("DROP TABLE IF EXISTS my_app.t2_test")
        config.session.execute("DROP TABLE IF EXISTS my_app.t2_test_0")
        config.session.execute("DROP TABLE IF EXISTS my_app.t2_test_1")
        config.session.execute("DROP TABLE IF EXISTS my_app.t2_test_2")
        so = TestStorageObj("t2")
        self.assertEquals('t2_test', so.test._table)
        so.test = {}
        self.assertEquals('t2_test_0', so.test._table)
        so.test = {1: 'a', 2: 'b'}
        self.assertEquals('t2_test_1', so.test._table)
        so.test = {3: 'c', 4: 'd'}
        self.assertEquals('t2_test_2', so.test._table)

    def test_storageobj_coherence_basic(self):
        '''
        test that two StorageObjs pointing to the same table work correctly.
        Changing data on one StorageObj is reflected on the other StorageObj.
        '''
        config.session.execute("DROP TABLE IF EXISTS my_app.test")
        so = Test2StorageObj('test')
        so.name = 'Oliver'
        so.age = 21
        so2 = Test2StorageObj('test')
        self.assertEqual(so.name, so2.name)
        self.assertEqual(so.age, so2.age)
        so.name = 'Benji'
        so2 = Test2StorageObj('test')
        self.assertEqual(so.name, so2.name)
        self.assertEqual(so.age, so2.age)
        config.session.execute("DROP TABLE IF EXISTS my_app.test")

    def test_storageobj_coherence_complex1(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.test")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso_0")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso2")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso2_test")
        so = Test3StorageObj('test')
        myso_attr = Test2StorageObj()
        myso_attr.name = 'Oliver'
        myso_attr.age = 21
        so.myso = myso_attr  # creates my_app.test_myso_0, the original attribute pointed to test_myso
        self.assertEqual(myso_attr.name, so.myso.name)
        del myso_attr
        self.assertEqual(so.myso.age, 21)
        config.session.execute("DROP TABLE IF EXISTS my_app.test")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso_0")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso2")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso2_test")

    def test_storageobj_coherence_complex2(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.test")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso_0")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso2")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso2_test")
        so = Test3StorageObj('test')
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
        config.session.execute("DROP TABLE IF EXISTS my_app.test")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso_0")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso2")
        config.session.execute("DROP TABLE IF EXISTS my_app.test_myso2_test")

    def test_get_attr_1(self):
        storage_obj = TestAttributes()
        storage_obj.do_nothing_at_all()
        value = 123
        storage_obj.setvalue(value)
        returned = storage_obj.getvalue()
        self.assertEqual(value, returned)

    def test_get_attr_2(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.test_attr")
        storage_obj = TestAttributes()
        storage_obj.do_nothing_at_all()
        value = 123
        storage_obj.setvalue(value)
        storage_obj.make_persistent("test_attr")
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

        config.session.execute("DROP TABLE IF EXISTS my_app.test_attr")

    def test_get_attr_3(self):
        # the same as test_get_attr_2 but the object is persistent since the beginning

        config.session.execute("DROP TABLE IF EXISTS my_app.test_attr")
        storage_obj = TestAttributes("test_attr")
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

        config.session.execute("DROP TABLE IF EXISTS my_app.test_attr")

    def test_recreation_init(self):
        """
        New StorageObj
        Persistent attributes
        Made persistent on the constructor.
        """
        sobj_name = "my_app.test_attr"
        config.session.execute("DROP TABLE IF EXISTS {}".format(sobj_name))
        attr1 = 'Test1'
        attr2 = 23
        storage_obj = Test2StorageObj(sobj_name)
        storage_obj.name = attr1
        storage_obj.age = attr2
        uuid_sobj = storage_obj.getID()

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
        """
        New StorageObj
        Has persistent and volatile attributes
        Made persistent on the constructor.
        """
        sobj_name = "my_app.test_attr"
        config.session.execute("DROP TABLE IF EXISTS {}".format(sobj_name))
        attr1 = 'Test1'
        attr2 = 23
        storage_obj = Test2StorageObj(sobj_name)
        storage_obj.name = attr1
        storage_obj.nonpersistent = attr2
        uuid_sobj = storage_obj.getID()

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
        """
        New StorageObj
        Persistent attributes
        Made persistent with make_persistent.
        """
        sobj_name = "my_app.test_attr"
        config.session.execute("DROP TABLE IF EXISTS {}".format(sobj_name))
        attr1 = 'Test1'
        attr2 = 23
        storage_obj = Test2StorageObj()
        storage_obj.make_persistent(sobj_name)
        uuid_sobj = storage_obj.getID()

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
        """
        New StorageObj
        Persistent attributes
        Made persistent with make_persistent.
        """
        sobj_name = "my_app.test_attr"
        config.session.execute("DROP TABLE IF EXISTS {}".format(sobj_name))
        attr1 = 'Test1'
        attr2 = 23
        storage_obj = Test2StorageObj()
        storage_obj.name = attr1
        storage_obj.volatile = 'Ofcourse'
        storage_obj.make_persistent(sobj_name)
        uuid_sobj = storage_obj.getID()

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
        sobj_name = "my_app.test_attr"
        config.session.execute("DROP TABLE IF EXISTS {}".format(sobj_name))
        config.session.execute("DROP TABLE IF EXISTS {}".format(sobj_name + '_myotherso'))
        config.session.execute("DROP TABLE IF EXISTS {}".format(sobj_name + '_myotherso_0'))

        storage_obj = Test2StorageObj()
        name_attr = 'Test1'
        age_attr = 23
        storage_obj.name = name_attr
        storage_obj.age = age_attr

        external_sobj = Test4StorageObj(sobj_name)
        external_sobj.myotherso = storage_obj

        uuid_sobj_internal = storage_obj.getID()
        uuid_sobj_external = external_sobj.getID()

        internal_name = external_sobj.myotherso._ksp + '.' + external_sobj.myotherso._table
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
            config.session.execute("SELECT * FROM {} WHERE storage_id={}".format(sobj_name, uuid_sobj_external)))

        try:
            result = result_set.next()
        except StopIteration as exc:
            self.fail("StorageObj istorage data was not saved")

        self.assertEqual(str(result.myotherso), uuid_sobj_internal)
        # They have been saved with the expected istorage ids

        external_sobj = Test4StorageObj(sobj_name)
        # Check internal configuration is correct
        self.assertEqual(external_sobj.getID(), uuid_sobj_external)
        self.assertEqual(external_sobj.myotherso.getID(), uuid_sobj_internal)
        self.assertEqual(external_sobj._ksp + '.' + external_sobj._table, sobj_name)
        self.assertEqual(external_sobj.myotherso._ksp + '.' + external_sobj.myotherso._table, internal_name)

        # Check data is correct
        self.assertEqual(external_sobj.myotherso.name, name_attr)
        self.assertEqual(external_sobj.myotherso.age, age_attr)

    def test_timestamp(self):
        config.session.execute("DROP TABLE IF EXISTS testing.time")
        a = TestTimestamp('testing.time')
        dt = datetime.fromtimestamp(1545730073)
        a.attr = dt.timestamp()

        del a
        import gc
        gc.collect()

        a = TestTimestamp('testing.time')
        self.assertEqual(a.attr, dt.timestamp())


if __name__ == '__main__':
    unittest.main()
