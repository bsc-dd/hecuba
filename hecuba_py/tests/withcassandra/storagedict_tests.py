import unittest
import uuid
import datetime
import time
import numpy as np
from random import randint

from hecuba import config, StorageObj, StorageDict
from hecuba.IStorage import build_remotely
from ..app.words import Words


class MyStorageDict(StorageDict):
    '''
    @TypeSpec dict<<position:int>, val:int>
    '''
    pass


class MyStorageDict2(StorageDict):
    '''
    @TypeSpec dict<<position:int, position2:str>, val:int>
    '''
    pass


class MyStorageDict3(StorageDict):
    '''
    @TypeSpec dict<<key:str>, val:int>
    '''


class MyStorageObjC(StorageObj):
    '''
    @ClassField mona dict<<a:str>, b:int>
    '''


class MyStorageDictA(StorageDict):
    '''
    @TypeSpec dict<<a:str>, b:int>
    '''


class mydict(StorageDict):
    '''
    @TypeSpec dict<<key0:int>, val0:tests.withcassandra.storagedict_tests.myobj2>
    '''


class myobj2(StorageObj):
    '''
    @ClassField attr1 int
    @ClassField attr2 str
    '''


class DictWithTuples(StorageDict):
    '''
    @TypeSpec dict<<key:int>, val:tuple<int,int>>
    '''


class DictWithTuples2(StorageDict):
    '''
    @TypeSpec dict<<key0:tuple<int,int>, key1:int>, val:str>
    '''


class DictWithTuples3(StorageDict):
    '''
    @TypeSpec dict<<key:int>, val0:int, val1:tuple<long,int>, val2:str, val3:tuple<str,float>>
    '''


class MultiTuples(StorageDict):
    '''
    @TypeSpec dict<<time:int, lat:double, lon:double, ilev:int>, m_cloudfract:tuple<float,float,float,int>, m_humidity:tuple<float,float,float,int>, m_icewater:tuple<float,float,float,int>, m_liquidwate:tuple<float,float,float,int>, m_ozone:tuple<float,float,float,int>, m_pot_vorticit:tuple<float,float,float,int>, m_rain:tuple<float,float,float,int>, m_snow:tuple<float,float,float,int>>
    '''


class Test2StorageObj(StorageObj):
    '''
       @ClassField name str
       @ClassField age int
    '''
    pass


class TestDictOfStorageObj(StorageDict):
    '''
        @TypeSpec dict<<key0:int>, val:tests.withcassandra.storageobj_tests.Test2StorageObj>
    '''


class DictWithDates(StorageDict):
    '''
    @TypeSpec dict<<date1:date>, date4:date>
    '''


class DictWithTimes(StorageDict):
    '''
    @TypeSpec dict<<date1:time>, date4:time>
    '''


class DictWithDateTimes(StorageDict):
    '''
    @TypeSpec dict<<date1:datetime>, date4:datetime>
    '''


class DictWithDateTimes2(StorageDict):
    '''
    @TypeSpec dict<<k:int>, v:datetime>
    '''


class MyStorageDictB(StorageDict):
    '''
    @TypeSpec dict<<a:str, b:int>, c:int>
    '''

class TestStorageObjNumpy(StorageObj):
    '''
       @ClassField mynumpy numpy.ndarray
    '''
    pass

class TestStorageObjNumpyEtAl(StorageObj):
    '''
       @ClassField mynumpy numpy.ndarray
       @ClassField name str
       @ClassField age int
       @ClassField rec tests.withcassandra.storagedict_tests.TestStorageObjNumpy
    '''
    pass

class TestStorageDictRec1(StorageDict):
    '''
       @TypeSpec dict<<key:int>, value:tests.withcassandra.storagedict_tests.TestStorageObjNumpyEtAl>
    '''

class TestStorageDictRec2(StorageDict):
    '''
       @TypeSpec dict<<key:int>, mynumpy:numpy.ndarray,name:str,age:int,rec:tests.withcassandra.storagedict_tests.TestStorageObjNumpy>
    '''

class TestStorageDictUnnamed(StorageDict):
    '''
       @TypeSpec dict<<int>, str>
    '''

class TestStorageDictUnnamed2(StorageDict):
    '''
       @TypeSpec dict<<key:int>, str>
    '''

class TestStorageDictUnnamed3(StorageDict):
    '''
       @TypeSpec dict<<int>, value:str>
    '''

class TestStorageDictUnnamed4(StorageDict):
    '''
       @TypeSpec dict<<int>, str,int>
    '''

class StorageDictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = config.execution_name
        config.execution_name = "StorageDictTest".lower()
    @classmethod
    def tearDownClass(cls):
        #config.session.execute("DROP KEYSPACE IF EXISTS {}".format(config.execution_name), timeout=60)
        config.execution_name = cls.old

    # Create a new keyspace per test
    def setUp(self):
        self.current_ksp = config.execution_name
        pass

    def tearDown(self):
        pass

    def test_init_empty(self):
        table = "test_init_empty"
        tablename = self.current_ksp+'.'+table
        tokens = [(1, 2), (2, 3), (3, 4)]
        nopars = StorageDict(tablename,
                             [('position', 'int')],
                             [('value', 'int')],
                             tokens=tokens)
        self.assertEqual(table, nopars._table)
        self.assertEqual(self.current_ksp, nopars._ksp)

        nopars.sync() # Wait until the data is persisted

        res = config.session.execute(
            'SELECT storage_id, primary_keys, columns, class_name, name, tokens, istorage_props,indexed_on ' +
            'FROM hecuba.istorage WHERE storage_id = %s', [nopars.storage_id])[0]

        self.assertEqual(uuid.uuid5(uuid.NAMESPACE_DNS, tablename), nopars.storage_id)
        self.assertEqual(nopars.__class__.__module__, 'hecuba.hdict')
        self.assertEqual(nopars.__class__.__name__, 'StorageDict')

        rebuild = build_remotely(res._asdict())
        self.assertEqual(rebuild._built_remotely, True)
        self.assertEqual(table, rebuild._table)
        self.assertEqual(self.current_ksp, rebuild._ksp)
        self.assertEqual(uuid.uuid5(uuid.NAMESPACE_DNS, tablename), rebuild.storage_id)

        self.assertEqual(nopars.storage_id, rebuild.storage_id)
        rebuild.delete_persistent()

    def test_init_empty_def_keyspace(self):
        tablename = "test_init_empty_def_keyspace"
        tokens = [(1, 2), (2, 3), (3, 4)]
        nopars = StorageDict(tablename,
                             [('position', 'int')],
                             [('value', 'int')],
                             tokens=tokens)
        self.assertEqual(tablename, nopars._table)
        self.assertEqual(config.execution_name, nopars._ksp)

        nopars.sync() # Wait until the data is persisted

        res = config.session.execute(
            'SELECT storage_id, primary_keys, columns, class_name, name, tokens, istorage_props,indexed_on ' +
            'FROM hecuba.istorage WHERE storage_id = %s', [nopars.storage_id])[0]

        self.assertEqual(uuid.uuid5(uuid.NAMESPACE_DNS, config.execution_name + '.' + tablename), nopars.storage_id)
        self.assertEqual(nopars.__class__.__module__, 'hecuba.hdict')
        self.assertEqual(nopars.__class__.__name__, 'StorageDict')

        rebuild = build_remotely(res._asdict())
        self.assertEqual(rebuild._built_remotely, True)
        self.assertEqual(tablename, rebuild._table)
        self.assertEqual(config.execution_name, rebuild._ksp)
        self.assertEqual(uuid.uuid5(uuid.NAMESPACE_DNS, config.execution_name + '.' + tablename), rebuild.storage_id)

        self.assertEqual(nopars.storage_id, rebuild.storage_id)
        rebuild.delete_persistent()

    def test_simple_insertions(self):
        tablename = "test_simple_insertions"
        tokens = [(1, 2), (2, 3), (3, 4)]
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')],
                         tokens=tokens)

        for i in range(100):
            pd[i] = 'ciao' + str(i)
        del pd
        import gc
        gc.collect()
        count, = config.session.execute('SELECT count(*) FROM '+self.current_ksp+'.'+tablename)[0]
        self.assertEqual(count, 100)
        #clean up
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')],
                         tokens=tokens)
        pd.delete_persistent()


    def test_dict_print(self):
        tablename = "test_dict_print"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])

        self.assertEquals(pd.__repr__(), "")

        pd[0] = 'a'
        self.assertEquals(pd.__repr__(), "{0: 'a'}")

        pd[1] = 'b'
        self.assertEquals(pd.__repr__(), "{1: 'b', 0: 'a'}")

        for i in range(1100):
            pd[i] = str(i)
        self.assertEquals(pd.__repr__().count(':'), 1000)
        pd.delete_persistent()

    def test_get_strs(self):
        tablename = "test_get_strs"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        pd[0] = 'str1'
        self.assertEquals(pd[0], 'str1')
        pd.delete_persistent()

    def test_len_memory(self):
        ninserts = 1500
        tablename = "test_len_memory"
        nopars = Words()
        self.assertIsNone(nopars.storage_id)
        nopars.ciao = 1
        nopars.ciao2 = "1"
        nopars.ciao3 = [1, 2, 3]
        nopars.ciao4 = (1, 2, 3)
        for i in range(ninserts):
            nopars.words[i] = 'ciao' + str(i)

        self.assertEqual(len(nopars.words), ninserts)

        nopars.make_persistent(tablename)
        self.assertEqual(len(nopars.words), ninserts)
        nopars.delete_persistent()

    def test_len_persistent(self):
        ninserts = 1500
        tablename = "test_len_persistent"
        nopars = Words(tablename)
        nopars.ciao = 1
        nopars.ciao2 = "1"
        nopars.ciao3 = [1, 2, 3]
        nopars.ciao4 = (1, 2, 3)
        for i in range(ninserts):
            nopars.words[i] = 'ciao' + str(i)

        self.assertEqual(len(nopars.words), ninserts)

        nopars.sync() # Wait until the data is persisted

        rebuild = Words(tablename)
        self.assertEqual(len(rebuild.words), ninserts)
        rebuild.delete_persistent()

    def test_make_persistent(self):
        # FIXME: CHANGE THE GENERATION OF THE TABLE NAME FOR ATTRIBUTES IN STORAGE OBJECT
        nopars = Words()
        tablename = "test_make_persistent"
        self.assertIsNone(nopars.storage_id)
        nopars.ciao = 1
        nopars.ciao2 = "1"
        nopars.ciao3 = [1, 2, 3]
        nopars.ciao4 = (1, 2, 3)
        for i in range(10):
            nopars.words[i] = 'ciao' + str(i)

        #count, = config.session.execute(
        #    "SELECT count(*) FROM system_schema.tables WHERE keyspace_name = '"+self.current_ksp+"' and table_name = 'Words_words'")[0]
        #self.assertEqual(0, count)
        self.assertFalse(nopars._is_persistent)


        nopars.make_persistent(tablename)
        cass_tablename = nopars.words._table

        del nopars
        import gc
        gc.collect()
        count, = config.session.execute('SELECT count(*) FROM '+self.current_ksp+'.'+cass_tablename)[0]
        self.assertEqual(10, count)

    def test_none_value(self):
        tablename = "test_none_value"
        mydict = MyStorageDict(tablename)
        mydict[0] = None
        self.assertEqual(mydict[0], None)
        mydict.delete_persistent()

    def test_none_keys(self):
        tablename = "test_none_keys"
        mydict = MyStorageDict(tablename)

        def set_none_key():
            mydict[None] = 1

        self.assertRaises(TypeError, set_none_key)
        mydict.delete_persistent()

    def test_paranoid_setitem_nonpersistent(self):
        tablename = "test_prnoid_set_nonp"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        pd[0] = 'bla'
        self.assertEquals(pd[0], 'bla')

        def set_wrong_val_1():
            pd[0] = 1

        self.assertRaises(TypeError, set_wrong_val_1)

        def set_wrong_val_2():
            pd['bla'] = 'bla'

        self.assertRaises(TypeError, set_wrong_val_2)
        pd.delete_persistent()

    def test_paranoid_setitem_multiple_nonpersistent(self):
        tablename = "test_prnoid_set_m_nonp"
        pd = StorageDict(tablename,
                         [('position1', 'int'), ('position2', 'text')],
                         [('value1', 'text'), ('value2', 'int')])
        pd[0, 'pos1'] = ['bla', 1]
        self.assertEquals(pd[0, 'pos1'], ('bla', 1))

        def set_wrong_val_1():
            pd[0, 'pos1'] = [1, 'bla']

        self.assertRaises(TypeError, set_wrong_val_1)

        def set_wrong_val_2():
            pd['pos1', 0] = ['bla', 1]

        self.assertRaises(TypeError, set_wrong_val_2)
        pd.delete_persistent()

    def test_paranoid_setitem_persistent(self):
        tablename = "test_prnoid_set_p"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        pd[0] = 'bla'

        pd.sync() # Wait until the data is persisted

        result = config.session.execute('SELECT value FROM '+self.current_ksp+'.'+tablename +' WHERE position = 0')
        for row in result:
            self.assertEquals(row.value, 'bla')

        def set_wrong_val_test():
            pd[0] = 1

        self.assertRaises(TypeError, set_wrong_val_test)
        pd.delete_persistent()

    def test_paranoid_setitem_multiple_persistent(self):
        tablename = "test_prnoid_set_N_p"
        pd = StorageDict(tablename,
                         [('position1', 'int'), ('position2', 'text')],
                         [('value1', 'text'), ('value2', 'int')])
        pd[0, 'pos1'] = ['bla', 1]
        for result in pd.values():
            self.assertEquals(result.value1, 'bla')
            self.assertEquals(result.value2, 1)

        def set_wrong_val():
            pd[0, 'pos1'] = ['bla', 'bla1']

        self.assertRaises(TypeError, set_wrong_val)

        def set_wrong_key():
            pd['bla', 'pos1'] = ['bla', 1]

        self.assertRaises(TypeError, set_wrong_key)
        pd.delete_persistent()

    def test_paranoid_setitemdouble_persistent(self):
        tablename = "test_prnoid_set_2_p"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'double')])
        pd[0] = 2.0
        pd.sync() # Wait until the data is persisted

        result = config.session.execute('SELECT value FROM '+self.current_ksp+'.'+tablename+' WHERE position = 0')
        for row in result:
            self.assertEquals(row.value, 2.0)

        def set_wrong_val_test():
            pd[0] = 1

        set_wrong_val_test()
        pd.delete_persistent()

    def test_paranoid_setitemdouble_multiple_persistent(self):
        tablename = "test_prnoid_set_2_N_p"
        pd = StorageDict(tablename,
                         [('position1', 'int'), ('position2', 'text')],
                         [('value1', 'text'), ('value2', 'double')])
        pd[0, 'pos1'] = ['bla', 1.0]
        time.sleep(2)
        self.assertEquals(pd[0, 'pos1'], ('bla', 1.0))
        pd.delete_persistent()

    def test_empty_persistent(self):
        # FIXME: CHANGE THE GENERATION OF THE TABLE NAME FOR ATTRIBUTES IN STORAGE OBJECT
        tablename = "test_empty_persistent"
        so = Words()
        so.make_persistent(tablename)
        so.ciao = "an attribute"
        so.another = 123
        config.batch_size = 1
        config.cache_activated = False
        for i in range(10):
            so.words[i] = str.join(',', map(lambda a: "ciao", range(i)))

        tbl_name = so.words._table
        del so
        import gc
        gc.collect()
        count, = config.session.execute('SELECT count(*) FROM '+self.current_ksp+'.'+tbl_name)[0]
        self.assertEqual(10, count)

        so = Words(tablename)
        so.delete_persistent()

        def delete_already_deleted():
            so.words.delete_persistent()

        self.assertRaises(RuntimeError, delete_already_deleted)

        try:
            count, = config.session.execute('SELECT count(*) FROM '+self.current_ksp+'.'+tbl_name)[0]
            error = False
        except Exception as e:
            error = True
        self.assertEquals(True, error)

    def test_simple_items_test(self):
        tablename = "test_simple_items_test"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])

        what_should_be = {}
        for i in range(100):
            pd[i] = 'ciao' + str(i)
            what_should_be[i] = 'ciao' + str(i)
        del pd
        import gc
        gc.collect()
        count, = config.session.execute('SELECT count(*) FROM '+self.current_ksp+'.'+tablename)[0]
        self.assertEqual(count, 100)
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        count = 0
        res = {}
        for key, val in pd.items():
            res[key] = val
            count += 1
        self.assertEqual(count, 100)
        self.assertEqual(what_should_be, res)
        pd.delete_persistent()

    def test_simple_values_test(self):
        tablename = "test_simple_values_test"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])

        what_should_be = set()
        for i in range(100):
            pd[i] = 'ciao' + str(i)
            what_should_be.add('ciao' + str(i))
        del pd
        import gc
        gc.collect()
        count, = config.session.execute('SELECT count(*) FROM '+self.current_ksp+'.'+tablename)[0]

        self.assertEqual(count, 100)

        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        count = 0
        res = set()
        for val in pd.values():
            res.add(val)
            count += 1
        self.assertEqual(count, 100)
        self.assertEqual(what_should_be, res)
        pd.delete_persistent()

    def test_simple_keys_test(self):
        tablename = "test_simple_keys_test"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])

        what_should_be = set()
        for i in range(100):
            pd[i] = 'ciao' + str(i)
            what_should_be.add(i)
        del pd
        import gc
        gc.collect()
        count, = config.session.execute('SELECT count(*) FROM '+self.current_ksp+'.'+tablename)[0]
        self.assertEqual(count, 100)
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        count = 0
        res = set()
        for val in pd.keys():
            res.add(val)
            count += 1
        self.assertEqual(count, 100)
        self.assertEqual(what_should_be, res)
        pd.delete_persistent()

    def test_simple_contains(self):
        tablename = "test_simple_contains"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])

        for i in range(100):
            pd[i] = 'ciao' + str(i)
        del pd
        import gc
        gc.collect()
        count, = config.session.execute('SELECT count(*) FROM '+self.current_ksp+'.'+tablename)[0]
        self.assertEqual(count, 100)

        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        for i in range(100):
            self.assertTrue(i in pd)
        pd.delete_persistent()

    def test_deleteitem_nonpersistent(self):
        pd = StorageDict(None,
                         [('position', 'int')],
                         [('value', 'text')])
        pd[0] = 'to_delete'
        del pd[0]

        def del_val():
            val = pd[0]

        self.assertRaises(KeyError, del_val)

        pd = StorageDict(None,
                         [('position', 'text')],
                         [('value', 'int')])
        pd['pos0'] = 0
        del pd['pos0']

        def del_val():
            val = pd['pos0']

        self.assertRaises(KeyError, del_val)

    def test_deleteitem_persistent(self):
        tablename = "test_deleteitem_persistent_1"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        pd[0] = 'to_delete'
        del pd[0]

        def del_val():
            val = pd[0]

        self.assertRaises(KeyError, del_val)
        pd.delete_persistent()

        tablename = "test_deleteitem_persistent_2"
        pd = StorageDict(tablename,
                         [('position', 'text')],
                         [('value', 'int')])
        pd['pos1'] = 0
        del pd['pos1']

        def del_val():
            val = pd['pos1']

        self.assertRaises(KeyError, del_val)
        pd.delete_persistent()

    def test_delete_two_keys(self):
        tablename = "test_delete_two_keys"
        o = MyStorageDictB(tablename)
        o["0", 0] = 0
        o["1", 1] = 1

        del o["0", 0]

        self.assertEqual(o["1", 1], 1)
        self.assertEqual(o.get(("0", 0), None), None)
        o.delete_persistent()

    def test_composed_items_test(self):
        tablename = "test_composed_items_test"
        pd = StorageDict(tablename,
                         primary_keys=[('pid', 'int'), ('time', 'int')],
                         columns=[('value', 'text'), ('x', 'double'), ('y', 'double'), ('z', 'double')])

        what_should_be = {}
        for i in range(100):
            pd[i, i + 100] = ['ciao' + str(i), i * 0.1, i * 0.2, i * 0.3]
            what_should_be[i, i + 100] = ['ciao' + str(i), i * 0.1, i * 0.2, i * 0.3]

        del pd
        import gc
        gc.collect()
        count, = config.session.execute('SELECT count(*) FROM '+self.current_ksp+'.'+tablename)[0]
        self.assertEqual(count, 100)
        pd = StorageDict(tablename)

        count = 0
        res = {}
        for key, val in pd.items():
            res[key] = val
            count += 1
        self.assertEqual(count, 100)
        delta = 0.000001
        for i in range(100):
            a = what_should_be[i, i + 100]
            b = res[i, i + 100]
            self.assertEqual(a[0], b.value)
            self.assertAlmostEquals(a[1], b.x, delta=delta)
            self.assertAlmostEquals(a[2], b.y, delta=delta)
            self.assertAlmostEquals(a[3], b.z, delta=delta)
        pd.delete_persistent()

    @unittest.skip("DEPRECATED: Disable changing the schema")
    def test_composed_key_return_list_items_test(self):
        tablename = "test_comkey_ret_list"
        pd = StorageDict(tablename,
                         primary_keys=[('pid', 'int'), ('time', 'double')],
                         columns=[('value', 'text'), ('x', 'double'), ('y', 'double'), ('z', 'double')])

        what_should_be = {}
        for i in range(100):
            pd[i, i + 100.0] = ['ciao' + str(i), i * 0.1, i * 0.2, i * 0.3]
            what_should_be[i, i + 100.0] = ['ciao' + str(i), i * 0.1, i * 0.2, i * 0.3]

        del pd
        import gc
        gc.collect()
        count, = config.session.execute('SELECT count(*) FROM '+self.current_ksp+'.'+tablename)[0]
        self.assertEqual(count, 100)
        pd = StorageDict(tablename,
                         [('pid', 'int')],
                         [('time', 'double'), ('value', 'text'), ('x', 'double'), ('y', 'double'), ('z', 'double')])
        count = 0
        res = {}
        for key, val in pd.items():
            self.assertTrue(isinstance(key, int))
            self.assertTrue(isinstance(val[0], float))
            res[key] = val
            count += 1
        self.assertEqual(count, 100)
        # casting to avoid 1.0000001 float python problem
        data = set([(key, int(val.time), val.value, int(val.x), int(val.y), int(val.z)) for key, val in pd.items()])
        data2 = set([(key[0], int(key[1]), val[0], int(val[1]), int(val[2]), int(val[3])) for key, val in
                     what_should_be.items()])
        self.assertEqual(data, data2)
        pd.delete_persistent()

    def test_storagedict_newinterface_localmemory(self):
        tablename = "test_sd_iface_localmem"
        my_dict = MyStorageDict()
        my_dict[0] = 1
        error = False
        try:
            result = config.session.execute('SELECT * FROM '+self.current_ksp+'.'+tablename)[0]
        except Exception as e:
            error = True
        self.assertEquals(True, error)
        # yolandab: if error == False we should delete the table
        if not error:
            config.session.execute('DROP TABLE '+self.current_ksp+'.'+tablename)



    def test_storagedict_newinterface_memorytopersistent(self):
        tablename = "test_sd_iface_mem2pers"
        my_dict =MyStorageDict()
        my_dict[0] = 1

        error = False
        try:
            result = config.session.execute('SELECT * FROM '+self.current_ksp+'.'+tablename)[0]
        except Exception as e:
            error = True
        self.assertEquals(True, error)
        # yolandab: if error == False we should delete the table
        if not error:
            config.session.execute('DROP TABLE '+self.current_ksp+'.'+tablename)

        tablename = "test_sd_iface_mem2pers"
        my_dict.make_persistent(tablename)

        del my_dict
        import gc
        gc.collect()
        count, = config.session.execute('SELECT count(*) FROM '+self.current_ksp+'.'+tablename)[0]
        self.assertEquals(1, count)
        #clean up
        my_dict = MyStorageDict(tablename)
        my_dict.delete_persistent()

    def test_storagedict_newinterface_persistent(self):
        tablename = "test_sd_iface_p"
        my_dict = MyStorageDict()
        my_dict[0] = 1
        my_dict.make_persistent(tablename)
        my_dict.sync() # Wait until the data is persisted

        count, = config.session.execute('SELECT count(*) FROM '+self.current_ksp+'.'+tablename)[0]
        self.assertEquals(1, count)

        my_dict[1] = 2
        my_dict.sync() # Wait until the data is persisted

        count, = config.session.execute('SELECT count(*) FROM '+self.current_ksp+'.'+tablename)[0]
        self.assertEquals(2, count)

        my_dict2 = MyStorageDict(tablename)
        self.assertEquals(1, my_dict2[0])
        self.assertEquals(2, my_dict2[1])
        my_dict2.delete_persistent()

    def test_update(self):
        tablename = "test_update1"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        pd[0] = 'prev_a'
        pd[1] = 'prev_b'
        self.assertEquals(pd[0], 'prev_a')
        self.assertEquals(pd[1], 'prev_b')
        pd.update({0: 'a', 1: 'b'})
        time.sleep(1)
        self.assertEquals(pd[0], 'a')
        self.assertEquals(pd[1], 'b')
        pd.update({2: 'c', 3: 'd'})
        time.sleep(1)
        self.assertEquals(pd[0], 'a')
        self.assertEquals(pd[1], 'b')
        self.assertEquals(pd[2], 'c')
        self.assertEquals(pd[3], 'd')
        tablename = "test_update2"
        pd2 = StorageDict(tablename,
                          [('position', 'int')],
                          [('value', 'text')])
        pd2[0] = 'final_a'
        pd2[4] = 'final_4'
        pd.update(pd2)
        time.sleep(1)
        self.assertEquals(pd[0], 'final_a')
        self.assertEquals(pd[4], 'final_4')
        pd.delete_persistent()
        pd2.delete_persistent()

    def test_update_kwargs(self):
        tablename = "test_update_kwargs"
        pd = StorageDict(tablename,
                         [('position', 'text')],
                         [('value', 'text')])
        pd['val1'] = 'old_a'
        pd['val2'] = 'old_b'
        time.sleep(2)
        self.assertEquals(pd['val1'], 'old_a')
        self.assertEquals(pd['val2'], 'old_b')
        pd.update(val1='new_a', val2='new_b')
        time.sleep(2)
        self.assertEquals(pd['val1'], 'new_a')
        self.assertEquals(pd['val2'], 'new_b')
        pd.delete_persistent()

    def test_get_persistent(self):
        table_name = 'test_get_persistent'
        my_text = MyStorageDict3(self.current_ksp + '.' + table_name)
        self.assertEquals(0, my_text.get('word', 0))
        my_text['word'] = my_text.get('word', 0) + 1
        time.sleep(2)
        self.assertEquals(1, my_text.get('word', 0))
        my_text.delete_persistent()

    def test_get_notpersistent(self):
        my_text = MyStorageDict3()
        self.assertEquals(0, my_text.get('word', 0))
        my_text['word'] = my_text.get('word', 0) + 1
        time.sleep(2)
        self.assertEquals(1, my_text.get('word', 0))

    def test_keys(self):
        my_dict = MyStorageDict2('test_keys')
        # int,text - int
        nitems = 100
        # write nitems to the dict
        for id in range(0, nitems):
            text_id = 'someText'
            # force some clash on second keys
            if id % 2 == 0:
                text_id = 'someText' + str(id)
            my_dict[(id, text_id)] = id

        my_dict.sync() # Wait until the data is persisted

        my_dict = MyStorageDict2('test_keys')
        total_items = list(my_dict.items())

        self.assertEqual(len(total_items), nitems)

        # del my_dict

        my_second_dict = MyStorageDict2()

        for id in range(nitems, 2 * nitems):
            text_id = 'someText'
            # force some clash on second keys
            if id % 2 == 0:
                text_id = 'someText' + str(id)
            my_second_dict[(id, text_id)] = id

        my_second_dict.make_persistent('test_keys')
        my_second_dict.sync() # Wait until the data is persisted

        my_second_dict = MyStorageDict2()
        my_second_dict.make_persistent('test_keys')

        total_items = list(my_second_dict.items())
        self.assertEqual(len(total_items), 2 * nitems)
        del my_dict
        del my_second_dict

        my_third_dict = MyStorageDict2('test_keys')
        total_items = list(my_third_dict.items())
        self.assertEqual(len(total_items), 2 * nitems)

        my_third_dict.delete_persistent()
        del my_third_dict

    def test_values(self):
        my_dict = MyStorageDict2('test_values')
        # int,text - int
        nitems = 100
        # write nitems to the dict
        for id in range(0, nitems):
            text_id = 'someText'
            # force some clash on second keys
            if id % 2 == 0:
                text_id = 'someText' + str(id)
            my_dict[(id, text_id)] = id

        my_dict.sync() # Wait until the data is persisted

        my_dict = MyStorageDict2('test_values')
        total_items = my_dict.items()

        self.assertEqual(len(list(total_items)), nitems)

        # del my_dict

        my_second_dict = MyStorageDict2()

        for id in range(nitems, 2 * nitems):
            text_id = 'someText'
            # force some clash on second keys
            if id % 2 == 0:
                text_id = 'someText' + str(id)
            my_second_dict[(id, text_id)] = id

        my_second_dict.make_persistent('test_values')
        my_second_dict.sync() # Wait until the data is persisted

        my_second_dict = MyStorageDict2()
        my_second_dict.make_persistent('test_values')

        total_items = list(my_second_dict.items())
        self.assertEqual(len(total_items), 2 * nitems)
        del my_dict
        del my_second_dict

        my_third_dict = MyStorageDict2('test_values')
        total_items = list(my_third_dict.items())
        self.assertEqual(len(total_items), 2 * nitems)

        my_third_dict.delete_persistent()
        del my_third_dict

    def test_items(self):
        my_dict = MyStorageDict2('test_items')
        # int,text - int
        nitems = 100
        # write nitems to the dict
        for id in range(0, nitems):
            text_id = 'someText'
            # force some clash on second keys
            if id % 2 == 0:
                text_id = 'someText' + str(id)
            my_dict[(id, text_id)] = id

        my_dict.sync() # Wait until the data is persisted

        my_dict = MyStorageDict2('test_items')
        total_items = list(my_dict.items())

        self.assertEqual(len(total_items), nitems)

        # del my_dict

        my_second_dict = MyStorageDict2()

        for id in range(nitems, 2 * nitems):
            text_id = 'someText'
            # force some clash on second keys
            if id % 2 == 0:
                text_id = 'someText' + str(id)
            my_second_dict[(id, text_id)] = id

        my_second_dict.make_persistent('test_items')
        my_second_dict.sync() # Wait until the data is persisted

        my_second_dict = MyStorageDict2()
        my_second_dict.make_persistent('test_items')

        total_items = list(my_second_dict.items())
        self.assertEqual(len(total_items), 2 * nitems)
        del my_dict
        del my_second_dict

        my_third_dict = MyStorageDict2('test_items')
        total_items = list(my_third_dict.items())
        self.assertEqual(len(total_items), 2 * nitems)

        my_third_dict.delete_persistent()
        del my_third_dict

    def test_iterator_sync(self):
        #'''
        #check that the prefetch returns the exact same number of elements as inserted
        #'''
        my_dict = MyStorageDict2('test_iterator_sync')
        # int,text - int
        nitems = 5000
        # write nitems to the dict
        for id in range(0, nitems):
            text_id = 'someText'
            # force some clash on second keys
            if id % 2 == 0:
                text_id = 'someText' + str(id)
            my_dict[(id, text_id)] = id

        total_items = list(my_dict.items())

        self.assertEqual(len(total_items), nitems)
        my_dict.delete_persistent()
        del my_dict

    def test_assign_and_replace(self):

        first_storagedict = MyStorageDictA()
        my_storageobj = MyStorageObjC("test_assign_and_replace1")
        self.assertIsNotNone(my_storageobj.mona.storage_id)
        self.assertTrue(isinstance(my_storageobj.mona.storage_id, uuid.UUID))

        # Creates the 'my_app.mystorageobjc_mona' table
        my_storageobj.mona['uno'] = 123

        # empty dict no persistent assigned to persistent object
        # creates the 'my_app.mystorageobjc_mona_0' table
        my_storageobj.mona = first_storagedict

        self.assertIsNotNone(my_storageobj.mona.storage_id)
        self.assertTrue(isinstance(my_storageobj.mona.storage_id, uuid.UUID))
        nitems = list(my_storageobj.mona.items())
        self.assertEqual(len(nitems), 0)
        # it was assigned to a persistent storage obj, it should be persistent
        self.assertIsNotNone(first_storagedict.storage_id)
        self.assertTrue(isinstance(first_storagedict.storage_id, uuid.UUID))
        # create another non persistent dict
        my_storagedict = MyStorageDictA()
        my_storagedict['due'] = 12341321
        # store the second non persistent dict into the StorageObj attribute
        my_storageobj.mona = my_storagedict
        # contents should not be merged, the contents should be the same as in the last storage_dict
        elements = list(my_storageobj.mona.items())
        self.assertEqual(len(elements), 1)
        my_storagedict = MyStorageDictA('test_assign_and_replace2')
        last_key = 'some_key'
        last_value = 123

        my_storagedict[last_key] = last_value
        # my_storageobj.mona
        my_storageobj.mona = my_storagedict
        self.assertTrue(last_key in my_storageobj.mona)

        last_items = list(my_storageobj.mona.items())
        self.assertEqual(len(last_items), 1)
        self.assertEqual(my_storagedict[last_key], last_value)

        my_storageobj.delete_persistent()

    def test_make_persistent_with_persistent_obj(self):
        o2 = myobj2("test_mkp_with_obj")
        o2.attr1 = 1
        o2.attr2 = "2"

        d = mydict()
        d[0] = o2
        try:
            d.make_persistent("test_mkp_dict_w_op")
        except Exception as ex:
            self.fail("Raised exception unexpectedly.\n" + str(ex))
        o2.delete_persistent()

    def test_int_tuples(self):
        d = DictWithTuples("test_int_tuples")

        what_should_be = dict()
        for i in range(0, 10):
            what_should_be[i] = (i, i + 10)
            d[i] = (i, i + 10)

        time.sleep(1)
        for i in range(0, 10):
            self.assertEqual(d[i], (i, i + 10))

        self.assertEqual(len(list(d.keys())), 10)

        res = dict()
        count = 0
        for key, item in d.items():
            res[key] = item
            count += 1

        self.assertEqual(count, len(what_should_be))
        self.assertEqual(what_should_be, res)
        d.delete_persistent()

    def test_values_tuples(self):
        # @TypeSpec dict<<key:int>, val0:int, val1:tuple<long,int>, val2:str, val3:tuple<str,float>>
        d = DictWithTuples3("test_values_tuples")

        what_should_be = set()
        for i in range(0, 20):
            what_should_be.add((i, (5500000000000000, i + 10), "hola", ("adios", (i + 20.5))))
            d[i] = [i, (5500000000000000, i + 10), "hola", ("adios", (i + 20.5))]

        time.sleep(1)
        res = set()
        count = 0
        for item in d.values():
            res.add(tuple(item))
            count += 1

        self.assertEqual(count, len(what_should_be))
        self.assertEqual(what_should_be, res)
        self.assertEqual(what_should_be, res)
        d.delete_persistent()

    def test_tuples_in_key(self):
        d = DictWithTuples2("test_tuples_in_key")

        for i in range(0, 10):
            d[(i, i), i + 1] = str(i)

        time.sleep(1)
        for i in range(0, 10):
            self.assertEqual(d[(i, i), i + 1], str(i))

        self.assertEqual(len(list(d.keys())), 10)
        d.delete_persistent()

    def test_keys_tuples(self):
        d = DictWithTuples2("test_keys_tuples")

        what_should_be = set()
        for i in range(0, 10):
            what_should_be.add(((i, i), i + 1))
            d[(i, i), i + 1] = str(i)

        time.sleep(1)

        res = set()
        count = 0
        for key in d.keys():
            res.add(tuple(key))
            count += 1

        self.assertEqual(count, len(what_should_be))
        self.assertEqual(what_should_be, res)
        d.delete_persistent()

    def test_multiple_tuples(self):
        d = DictWithTuples3("test_multiple_tuples")

        what_should_be = dict()
        for i in range(0, 10):
            what_should_be[i] = [i, (5500000000000000, i + 10), "hola", ("adios", (i + 20.5))]
            d[i] = [i, (5500000000000000, i + 10), "hola", ("adios", (i + 20.5))]

        time.sleep(2)
        for i in range(0, 10):
            self.assertEqual(list(d[i]), [i, (5500000000000000, i + 10), "hola", ("adios", (i + 20.5))])
        self.assertEqual(len(list(d.keys())), 10)

        res = dict()
        count = 0
        for key, item in d.items():
            res[key] = list(item)
            count += 1

        self.assertEqual(count, len(what_should_be))
        self.assertEqual(what_should_be, res)
        d.delete_persistent()

    def test_int_tuples_null_values(self):
        d = DictWithTuples("test_int_tuples_null_values")

        for i in range(0, 10):
            if i % 2 == 0:
                d[i] = (None, i + 10)
            else:
                d[i] = (i, i + 10)

        d.sync() # Wait until the data is persisted

        d = DictWithTuples("test_int_tuples_null_values")
        for i in range(0, 10):
            if i % 2 == 0:
                self.assertEqual(d[i], (None, i + 10))
            else:
                self.assertEqual(d[i], (i, i + 10))
        d.delete_persistent()

    def test_multi_tuples(self):
        d = MultiTuples("test_multi_tuples")
        what_should_be = dict()

        for i in range(0, 10):
            d[(i, i, i, i)] = [(i, i, i, i), (i, i, i, i), (i, i, i, i), (i, i, i, i), (i, i, i, i), (i, i, i, i),
                               (i, i, i, i), (i, i, i, i)]
            what_should_be[(i, i, i, i)] = [(i, i, i, i), (i, i, i, i), (i, i, i, i), (i, i, i, i), (i, i, i, i),
                                            (i, i, i, i),
                                            (i, i, i, i), (i, i, i, i)]
        for i in range(0, 10):
            self.assertEqual(list(d[(i, i, i, i)]),
                             [(float(i), float(i), float(i), i), (float(i), float(i), float(i), i),
                              (float(i), float(i), float(i), i), (float(i), float(i), float(i), i),
                              (float(i), float(i), float(i), i), (float(i), float(i), float(i), i),
                              (float(i), float(i), float(i), i), (float(i), float(i), float(i), i)])
        d.delete_persistent()

    def test_multiple_tuples_NULL(self):
        d = DictWithTuples3("test_multiple_tuples_NULL")

        what_should_be = dict()
        for i in range(0, 10):
            if i % 2 == 0:
                what_should_be[i] = [i, (5500000000000000, i + 10), "hola", ("adios", (i + 20.5))]
                d[i] = [i, (5500000000000000, i + 10), "hola", ("adios", (i + 20.5))]
            else:
                what_should_be[i] = [i, (5500000000000000, None), "hola", (None, (i + 20.5))]
                d[i] = [i, (5500000000000000, None), "hola", (None, (i + 20.5))]

        d.sync() # Wait until the data is persisted

        d = DictWithTuples3("test_multiple_tuples_NULL")
        for i in range(0, 10):
            if i % 2 == 0:
                self.assertEqual(list(d[i]), [i, (5500000000000000, i + 10), "hola", ("adios", (i + 20.5))])
            else:
                self.assertEqual(list(d[i]), [i, (5500000000000000, None), "hola", (None, (i + 20.5))])

        self.assertEqual(len(list(d.keys())), 10)

        res = dict()
        count = 0
        for key, item in d.items():
            res[key] = list(item)
            count += 1

        self.assertEqual(count, len(what_should_be))
        self.assertEqual(what_should_be, res)
        d.delete_persistent()

    def test_storagedict_objs_same_table(self):
        d = TestDictOfStorageObj("test_storagedict_objs_same_table")
        for i in range(0, 10):
            o = Test2StorageObj()
            o.name = "adri" + str(i)
            o.age = i
            d[i] = o

        n = len(d)
        for i in range(0, n):
            self.assertEqual(d[i]._ksp.lower(), self.current_ksp)
            self.assertEqual(d[i]._table.lower(), "test2storageobj")
        d.delete_persistent()

    def gen_random_date(self):
        return datetime.date(year=randint(2000, 2019), month=randint(1, 12), day=randint(1, 28))

    def gen_random_datetime(self):
        return datetime.datetime(year=randint(2000, 2019), month=randint(1, 12), day=randint(1, 28),
                                 hour=randint(0, 23), minute=randint(0, 59), second=randint(0, 59))

    def gen_random_time(self):
        return datetime.time(hour=randint(0, 23), minute=randint(0, 59), second=randint(0, 59),
                             microsecond=randint(0, 59))

    def test_multiple_dates(self):
        d = DictWithDates("test_multiple_dates")
        what_should_be = dict()
        for i in range(0, 50):
            keys = self.gen_random_date()
            cols = self.gen_random_date()
            what_should_be[keys] = [cols]
            d[keys] = [cols]

        d.sync() # Wait until the data is persisted

        d = DictWithDates("test_multiple_dates")

        self.assertEqual(len(list(d.keys())), len(what_should_be.keys()))

        count = 0
        for k in what_should_be.keys():
            count += 1
            self.assertEqual(what_should_be[k], [d[k]])

        self.assertEqual(count, len(list(d)))
        d.delete_persistent()

    def test_multiple_times(self):
        d = DictWithTimes("test_multiple_times")
        what_should_be = dict()
        for i in range(0, 50):
            keys = self.gen_random_time()
            cols = self.gen_random_time()
            what_should_be[keys] = [cols]
            d[keys] = [cols]

        d.sync() # Wait until the data is persisted

        d = DictWithTimes("test_multiple_times")

        self.assertEqual(len(list(d.keys())), len(what_should_be.keys()))

        count = 0
        for k in what_should_be.keys():
            count += 1
            self.assertEqual(what_should_be[k], [d[k]])

        self.assertEqual(count, len(list(d)))
        d.delete_persistent()

    def test_datetimes(self):
        d = DictWithDateTimes("test_datetimes")
        what_should_be = dict()
        for i in range(0, 50):
            keys = self.gen_random_datetime()
            cols = self.gen_random_datetime()
            what_should_be[keys] = [cols]
            d[keys] = [cols]

        d.sync() # Wait until the data is persisted

        d = DictWithDateTimes("test_datetimes")
        self.assertEqual(len(list(d.keys())), len(what_should_be.keys()))
        count = 0
        for k in what_should_be.keys():
            count += 1
            self.assertEqual(what_should_be[k], [d[k]])

        self.assertEqual(count, len(list(d)))
        d.delete_persistent()

    def test_sync(self):
        myo = TestStorageObjNumpyEtAl()
        myo.mynumpy = np.arange(22*22).reshape(22,22)
        myo.name = "uyuyuy"
        myo.age = 42
        myo.rec.mynumpy = np.arange(20*20).reshape(20,20)
        myo.dummy = "Whatever"

        myd = TestStorageDictRec1("test_sync")

        # WARNING: We keep each dictionary item reference into a list to AVOID
        # rebuilding them.  Otherwise, each 'myd[i]' DELETES the
        # previous instance.
        o = []
        for i in range(0,3):
            myd[i]=myo
            o.append(myd[i])

        ##print("sids myo={} myd[0]={} ({} myo refs)".format(getattr(myo, 'storage_id',None), o[0].storage_id, sys.getrefcount(myo)), flush=True)

        myd.sync() # Wait until the data is persisted

        del myo # Remove from memory
        del myd # Remove from memory
        myd = TestStorageDictRec1("test_sync")

        o = []
        for i in range(0,3):
            o.append(myd[i])

        for i in range(0,3):
            o[i].mynumpy[0,0] = -666 + i    # Asynchronous write

        x = TestStorageDictRec1("test_sync")

        for i in range(0,3):
            ##print("myd[{}].mynumpy={} x[{}].mynumpy={}".format(i,o[i].mynumpy.storage_id,i, x[i].mynumpy.storage_id),flush=True)
            ##print("myd[{}]: {}  x[{}]:{}".format(i,o[i].mynumpy[0,0], i, x[i].mynumpy[0,0]),flush=True)
            self.assertTrue(o[i].mynumpy[0,0] != x[i].mynumpy[0,0]) # Data should be still in dirty/flight WARNING! This makes the hypothesis that the time it takes for the writes is high enough to have time to instantiate with a previous value instead of the last one... depending on the environment this may NOT be true.

        myd.sync()
        ##print("AFTER SYNC2", flush=True)

        x = TestStorageDictRec1("test_sync")
        for i in range(0,3):
            self.assertTrue(myd[i].mynumpy[0,0] == x[i].mynumpy[0,0])


        for i in range(0,3):
            myd[i].rec.mynumpy[0,0] = -1666 + i    # Asynchronous write

        x = TestStorageDictRec1("test_sync")
        for i in range(0,3):
            ##print("myd[{}]: {}  x[{}]:{}".format(i,myd[i].rec.mynumpy[0,0], i, x[i].rec.mynumpy[0,0]),flush=True)
            self.assertTrue(myd[i].rec.mynumpy[0,0] == x[i].rec.mynumpy[0,0]) # Data is still in dirty/flight

        myd.sync()

        x = TestStorageDictRec1("test_sync")
        for i in range(0,3):
            self.assertTrue(myd[i].rec.mynumpy[0,0] == x[i].rec.mynumpy[0,0])

    def test_unnamed(self):
        myo = TestStorageDictUnnamed("test_unnamed")
        myo[42] = "hola"
        myo.sync()
        myo = TestStorageDictUnnamed("test_unnamed")
        self.assertTrue(myo[42] == "hola")

    def test_unnamed2(self):
        myo = TestStorageDictUnnamed2("test_unnamed2")
        myo[42] = "hola"
        myo.sync()
        myo = TestStorageDictUnnamed2("test_unnamed2")
        self.assertTrue(myo[42] == "hola")

    def test_unnamed3(self):
        myo = TestStorageDictUnnamed3("test_unnamed3")
        myo[43] = "hola"
        myo.sync()
        myo = TestStorageDictUnnamed3("test_unnamed3")
        self.assertTrue(myo[43] == "hola")

    def test_unnamed4(self):
        myo = TestStorageDictUnnamed4("test_unnamed4")
        myo[42] = ["hola", 666]
        myo.sync()
        myo = TestStorageDictUnnamed4("test_unnamed4")
        self.assertTrue(list(myo[42]), ["hola", 666])

    def test_make_persistent2(self):

        class MyStorageDictA1(StorageDict):
            '''
            @TypeSpec dict<<a:str,a2:int>, b:int>
            '''
        d = MyStorageDictA1("test_make_persistent2")
        d["hola",666] = 42

        d.sync()

        class MyStorageDictA2(StorageDict):
            '''
            @TypeSpec dict<<a:str>, b:int>
            '''
        d = MyStorageDictA2()  # VOLATILE
        d['uy'] = 666
        with self.assertRaises(RuntimeError) as context:
            d.make_persistent("test_make_persistent2") # SHOULD FAIL!!! DIFFERENT SCHEMA!!

    def test_store_persistent_numpy(self):
        from hecuba import StorageNumpy
        class MyStorageDictNumpy(StorageDict):
            '''
            @TypeSpec dict<<a:int>, b:numpy.ndarray, c: int>
            '''
        d = MyStorageDictNumpy("test_store_persistent_numpy")
        n = np.arange(3*4).reshape(3,4)
        s = StorageNumpy(n, "test_store_persistent_numpyELNUMPY")
        d[42] = [s, 666]
        self.assertEqual(d[42].b.storage_id,  s.storage_id)


if __name__ == '__main__':
    unittest.main()
