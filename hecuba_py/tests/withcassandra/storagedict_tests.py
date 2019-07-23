import time as user_time
import unittest
import uuid
from datetime import date, datetime, time
from random import randint

from hecuba import config, StorageObj, StorageDict
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


class DictWithDates(StorageDict):
    '''
    @TypeSpec dict<<date1:date>, date4:date>
    '''


class DictWithTimes(StorageDict):
    '''
    @TypeSpec dict<<date1:time>, date4:time>
    '''


class StorageDictTest(unittest.TestCase):
    def test_init_empty(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab1")
        tablename = "ksp.tab1"
        tokens = [(1, 2), (2, 3), (3, 4)]
        nopars = StorageDict(tablename,
                             [('position', 'int')],
                             [('value', 'int')],
                             tokens)
        self.assertEqual("tab1", nopars._table)
        self.assertEqual("ksp", nopars._ksp)

        res = config.session.execute(
            'SELECT storage_id, primary_keys, columns, class_name, name, tokens, istorage_props,indexed_on ' +
            'FROM hecuba.istorage WHERE storage_id = %s', [nopars._storage_id])[0]

        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, tablename), nopars._storage_id)
        self.assertEqual(nopars.__class__.__module__, 'hecuba.hdict')
        self.assertEqual(nopars.__class__.__name__, 'StorageDict')

        rebuild = StorageDict.build_remotely(res._asdict())
        self.assertEqual(rebuild._built_remotely, True)
        self.assertEqual('tab1', rebuild._table)
        self.assertEqual("ksp", rebuild._ksp)
        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, tablename), rebuild._storage_id)

        self.assertEqual(nopars._is_persistent, rebuild._is_persistent)

    def test_init_empty_def_keyspace(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab1")
        tablename = "tab1"
        tokens = [(1, 2), (2, 3), (3, 4)]
        nopars = StorageDict(tablename,
                             [('position', 'int')],
                             [('value', 'int')],
                             tokens)
        self.assertEqual("tab1", nopars._table)
        self.assertEqual(config.execution_name, nopars._ksp)

        res = config.session.execute(
            'SELECT storage_id, primary_keys, columns, class_name, name, tokens, istorage_props,indexed_on ' +
            'FROM hecuba.istorage WHERE storage_id = %s', [nopars._storage_id])[0]

        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.' + tablename), nopars._storage_id)
        self.assertEqual(nopars.__class__.__module__, 'hecuba.hdict')
        self.assertEqual(nopars.__class__.__name__, 'StorageDict')

        rebuild = StorageDict.build_remotely(res._asdict())
        self.assertEqual(rebuild._built_remotely, True)
        self.assertEqual('tab1', rebuild._table)
        self.assertEqual(config.execution_name, rebuild._ksp)
        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.' + tablename), rebuild._storage_id)

        self.assertEqual(nopars._is_persistent, rebuild._is_persistent)

    def test_simple_insertions(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab10")
        tablename = "tab10"
        tokens = [(1, 2), (2, 3), (3, 4)]
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')],
                         tokens)

        for i in range(100):
            pd[i] = 'ciao' + str(i)
        del pd
        count, = config.session.execute('SELECT count(*) FROM my_app.tab10')[0]
        self.assertEqual(count, 100)

    def test_dict_print(self):
        tablename = "tab10"
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
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

    def test_get_strs(self):
        tablename = "tab10"
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        pd[0] = 'str1'
        self.assertEquals(pd[0], 'str1')
        '''
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'list<text>')])
        pd[0] = ['str1', 'str2']
        self.assertEquals(pd[0], ['str1', 'str2'])

        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'tuple<text,text>')])
        pd[0] = 'str1', 'str2'
        self.assertEquals(pd[0], 'str1', 'str2')
        '''

    def test_make_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.t_make_words")
        nopars = Words()
        self.assertFalse(nopars._is_persistent)
        nopars.ciao = 1
        nopars.ciao2 = "1"
        nopars.ciao3 = [1, 2, 3]
        nopars.ciao4 = (1, 2, 3)
        for i in range(10):
            nopars.words[i] = 'ciao' + str(i)

        count, = config.session.execute(
            "SELECT count(*) FROM system_schema.tables WHERE keyspace_name = 'my_app' and table_name = 't_make_words'")[
            0]
        self.assertEqual(0, count)

        nopars.make_persistent("t_make")

        del nopars
        count, = config.session.execute('SELECT count(*) FROM my_app.t_make_words')[0]
        self.assertEqual(10, count)

    def test_none_value(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.somename")
        mydict = MyStorageDict('somename')
        mydict[0] = None
        self.assertEqual(mydict[0], None)
        config.session.execute("DROP TABLE IF EXISTS my_app.somename")

    def test_none_keys(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.somename")
        mydict = MyStorageDict('somename')

        def set_none_key():
            mydict[None] = 1

        self.assertRaises(TypeError, set_none_key)
        config.session.execute("DROP TABLE IF EXISTS my_app.somename")

    def test_paranoid_setitem_nonpersistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mydict")
        pd = StorageDict("mydict",
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

    def test_paranoid_setitem_multiple_nonpersistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mydict")
        pd = StorageDict("mydict",
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

    def test_paranoid_setitem_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab_a1")
        pd = StorageDict("tab_a1",
                         [('position', 'int')],
                         [('value', 'text')])
        pd[0] = 'bla'
        result = config.session.execute('SELECT value FROM my_app.tab_a1 WHERE position = 0')
        for row in result:
            self.assertEquals(row.value, 'bla')

        def set_wrong_val_test():
            pd[0] = 1

        self.assertRaises(TypeError, set_wrong_val_test)

    def test_paranoid_setitem_multiple_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab_a2")
        pd = StorageDict("tab_a2",
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

    def test_paranoid_setitemdouble_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab_a3")
        pd = StorageDict("tab_a3",
                         [('position', 'int')],
                         [('value', 'double')])
        pd[0] = 2.0
        result = config.session.execute('SELECT value FROM my_app.tab_a3 WHERE position = 0')
        for row in result:
            self.assertEquals(row.value, 2.0)

        def set_wrong_val_test():
            pd[0] = 1

        set_wrong_val_test()

    def test_paranoid_setitemdouble_multiple_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab_a4")
        pd = StorageDict("tab_a4",
                         [('position1', 'int'), ('position2', 'text')],
                         [('value1', 'text'), ('value2', 'double')])
        pd[0, 'pos1'] = ['bla', 1.0]
        user_time.sleep(2)
        self.assertEquals(pd[0, 'pos1'], ('bla', 1.0))

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

        so = Words("wordsso")
        so.delete_persistent()
        so.words.delete_persistent()

        count, = config.session.execute('SELECT count(*) FROM my_app.wordsso_words')[0]
        self.assertEqual(0, count)

    def test_simple_items_test(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab_a1")

        pd = StorageDict("tab_a1",
                         [('position', 'int')],
                         [('value', 'text')])

        what_should_be = {}
        for i in range(100):
            pd[i] = 'ciao' + str(i)
            what_should_be[i] = 'ciao' + str(i)
        del pd
        count, = config.session.execute('SELECT count(*) FROM my_app.tab_a1')[0]
        self.assertEqual(count, 100)
        pd = StorageDict("tab_a1",
                         [('position', 'int')],
                         [('value', 'text')])
        count = 0
        res = {}
        for key, val in pd.items():
            res[key] = val
            count += 1
        self.assertEqual(count, 100)
        self.assertEqual(what_should_be, res)

    def test_simple_values_test(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab_a2")
        tablename = "tab_a2"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])

        what_should_be = set()
        for i in range(100):
            pd[i] = 'ciao' + str(i)
            what_should_be.add('ciao' + str(i))
        del pd
        count, = config.session.execute('SELECT count(*) FROM my_app.tab_a2')[0]

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

    def test_simple_keys_test(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab_a3")
        tablename = "tab_a3"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])

        what_should_be = set()
        for i in range(100):
            pd[i] = 'ciao' + str(i)
            what_should_be.add(i)
        del pd
        count, = config.session.execute('SELECT count(*) FROM my_app.tab_a3')[0]
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

    def test_simple_contains(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab_a4")
        tablename = "tab_a4"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])

        for i in range(100):
            pd[i] = 'ciao' + str(i)
        del pd
        count, = config.session.execute('SELECT count(*) FROM my_app.tab_a4')[0]
        self.assertEqual(count, 100)

        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        for i in range(100):
            self.assertTrue(i in pd)

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
        tablename = "tab_a5"
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        pd[0] = 'to_delete'
        del pd[0]

        def del_val():
            val = pd[0]

        self.assertRaises(KeyError, del_val)

        tablename = "tab_a6"
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        pd = StorageDict(tablename,
                         [('position', 'text')],
                         [('value', 'int')])
        pd['pos1'] = 0
        del pd['pos1']

        def del_val():
            val = pd['pos1']

        self.assertRaises(KeyError, del_val)

    def test_composed_items_test(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab12")
        tablename = "tab12"
        pd = StorageDict(tablename,
                         [('pid', 'int'), ('time', 'int')],
                         [('value', 'text'), ('x', 'double'), ('y', 'double'), ('z', 'double')])

        what_should_be = {}
        for i in range(100):
            pd[i, i + 100] = ['ciao' + str(i), i * 0.1, i * 0.2, i * 0.3]
            what_should_be[i, i + 100] = ['ciao' + str(i), i * 0.1, i * 0.2, i * 0.3]

        del pd

        count, = config.session.execute('SELECT count(*) FROM my_app.tab12')[0]
        self.assertEqual(count, 100)
        pd = StorageDict(tablename,
                         [('pid', 'int'), ('time', 'int')],
                         [('value', 'text'), ('x', 'double'), ('y', 'double'), ('z', 'double')])
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

    def test_composed_key_return_list_items_test(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab13")
        tablename = "tab13"
        pd = StorageDict(tablename,
                         [('pid', 'int'), ('time', 'double')],
                         [('value', 'text'), ('x', 'double'), ('y', 'double'), ('z', 'double')])

        what_should_be = {}
        for i in range(100):
            pd[i, i + 100.0] = ['ciao' + str(i), i * 0.1, i * 0.2, i * 0.3]
            what_should_be[i, i + 100.0] = ['ciao' + str(i), i * 0.1, i * 0.2, i * 0.3]

        del pd

        count, = config.session.execute('SELECT count(*) FROM my_app.tab13')[0]
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

    def test_storagedict_newinterface_localmemory(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.my_dict")

        my_dict = MyStorageDict()
        my_dict[0] = 1
        error = False
        try:
            result = config.session.execute('SELECT * FROM my_app.my_dict')[0]
        except Exception as e:
            error = True
        self.assertEquals(True, error)

    def test_storagedict_newinterface_memorytopersistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.my_dict")

        my_dict = MyStorageDict()
        my_dict[0] = 1
        error = False
        try:
            result = config.session.execute('SELECT * FROM my_app.my_dict')[0]
        except Exception as e:
            error = True
        self.assertEquals(True, error)

        my_dict.make_persistent('my_dict')

        del my_dict

        count, = config.session.execute('SELECT count(*) FROM my_app.my_dict')[0]
        self.assertEquals(1, count)

    def test_storagedict_newinterface_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.my_dict")

        my_dict = MyStorageDict()
        my_dict[0] = 1
        my_dict.make_persistent('my_dict')
        user_time.sleep(1)
        count, = config.session.execute('SELECT count(*) FROM my_app.my_dict')[0]
        self.assertEquals(1, count)

        my_dict[1] = 2
        user_time.sleep(1)
        count, = config.session.execute('SELECT count(*) FROM my_app.my_dict')[0]
        self.assertEquals(2, count)

        my_dict2 = MyStorageDict('my_dict')
        self.assertEquals(1, my_dict2[0])
        self.assertEquals(2, my_dict2[1])

    def test_update(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab_a4")
        config.session.execute("DROP TABLE IF EXISTS my_app.tab_a5")
        tablename = "tab_a4"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        pd[0] = 'prev_a'
        pd[1] = 'prev_b'
        self.assertEquals(pd[0], 'prev_a')
        self.assertEquals(pd[1], 'prev_b')
        pd.update({0: 'a', 1: 'b'})
        user_time.sleep(1)
        self.assertEquals(pd[0], 'a')
        self.assertEquals(pd[1], 'b')
        pd.update({2: 'c', 3: 'd'})
        user_time.sleep(1)
        self.assertEquals(pd[0], 'a')
        self.assertEquals(pd[1], 'b')
        self.assertEquals(pd[2], 'c')
        self.assertEquals(pd[3], 'd')
        tablename = "tab_a5"
        pd2 = StorageDict(tablename,
                          [('position', 'int')],
                          [('value', 'text')])
        pd2[0] = 'final_a'
        pd2[4] = 'final_4'
        pd.update(pd2)
        user_time.sleep(1)
        self.assertEquals(pd[0], 'final_a')
        self.assertEquals(pd[4], 'final_4')

    def test_update_kwargs(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab_a6")
        tablename = "tab_a6"
        pd = StorageDict(tablename,
                         [('position', 'text')],
                         [('value', 'text')])
        pd['val1'] = 'old_a'
        pd['val2'] = 'old_b'
        user_time.sleep(2)
        self.assertEquals(pd['val1'], 'old_a')
        self.assertEquals(pd['val2'], 'old_b')
        pd.update(val1='new_a', val2='new_b')
        user_time.sleep(2)
        self.assertEquals(pd['val1'], 'new_a')
        self.assertEquals(pd['val2'], 'new_b')

    def test_get_persistent(self):
        table_name = 'tab_a7'
        config.session.execute("DROP TABLE IF EXISTS my_app." + table_name)
        my_text = MyStorageDict3('my_app.' + table_name)
        self.assertEquals(0, my_text.get('word', 0))
        my_text['word'] = my_text.get('word', 0) + 1
        user_time.sleep(2)
        self.assertEquals(1, my_text.get('word', 0))

    def test_get_notpersistent(self):
        my_text = MyStorageDict3()
        self.assertEquals(0, my_text.get('word', 0))
        my_text['word'] = my_text.get('word', 0) + 1
        user_time.sleep(2)
        self.assertEquals(1, my_text.get('word', 0))

    def test_keys(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.test_keys")
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

        del my_dict  # force sync
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
        del my_second_dict  # force sync
        my_second_dict = MyStorageDict2()
        my_second_dict.make_persistent('test_keys')

        total_items = list(my_second_dict.items())
        self.assertEqual(len(total_items), 2 * nitems)
        del my_dict
        del my_second_dict

        my_third_dict = MyStorageDict2('test_keys')
        total_items = list(my_third_dict.items())
        self.assertEqual(len(total_items), 2 * nitems)

        del my_third_dict
        config.session.execute("DROP TABLE IF EXISTS my_app.test_keys")

    def test_values(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.test_values")
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

        del my_dict  # force sync
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
        del my_second_dict  # force sync
        my_second_dict = MyStorageDict2()
        my_second_dict.make_persistent('test_values')

        total_items = list(my_second_dict.items())
        self.assertEqual(len(total_items), 2 * nitems)
        del my_dict
        del my_second_dict

        my_third_dict = MyStorageDict2('test_values')
        total_items = list(my_third_dict.items())
        self.assertEqual(len(total_items), 2 * nitems)

        del my_third_dict
        config.session.execute("DROP TABLE IF EXISTS my_app.test_values")

    def test_items(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.test_items")
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

        del my_dict  # force sync
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
        del my_second_dict  # force sync
        my_second_dict = MyStorageDict2()
        my_second_dict.make_persistent('test_items')

        total_items = list(my_second_dict.items())
        self.assertEqual(len(total_items), 2 * nitems)
        del my_dict
        del my_second_dict

        my_third_dict = MyStorageDict2('test_items')
        total_items = list(my_third_dict.items())
        self.assertEqual(len(total_items), 2 * nitems)

        del my_third_dict
        config.session.execute("DROP TABLE IF EXISTS my_app.test_items")

    def test_iterator_sync(self):
        '''
        check that the prefetch returns the exact same number of elements as inserted 
        '''
        config.session.execute("DROP TABLE IF EXISTS my_app.test_iterator_sync")
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
        del my_dict
        config.session.execute("DROP TABLE IF EXISTS my_app.test_iterator_sync")

    def test_assign_and_replace(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.first_name")
        config.session.execute("DROP TABLE IF EXISTS my_app.first_name_mona")
        config.session.execute("DROP TABLE IF EXISTS my_app.first_name_mona_0")
        config.session.execute("DROP TABLE IF EXISTS my_app.first_name_mona_1")
        config.session.execute("DROP TABLE IF EXISTS my_app.second_name")

        first_storagedict = MyStorageDictA()
        my_storageobj = MyStorageObjC("first_name")
        self.assertTrue(my_storageobj.mona._is_persistent)

        # Creates the 'my_app.first_name_mona' table
        my_storageobj.mona['uno'] = 123

        # empty dict no persistent assigned to persistent object
        # creates the 'my_app.first_name_mona_0' table
        my_storageobj.mona = first_storagedict

        self.assertTrue(my_storageobj.mona._is_persistent)
        nitems = list(my_storageobj.mona.items())
        self.assertEqual(len(nitems), 0)
        # it was assigned to a persistent storage obj, it should be persistent
        self.assertTrue(first_storagedict._is_persistent)
        # create another non persistent dict
        my_storagedict = MyStorageDictA()
        my_storagedict['due'] = 12341321
        # store the second non persistent dict into the StorageObj attribute
        my_storageobj.mona = my_storagedict
        # contents should not be merged, the contents should be the same as in the last storage_dict
        elements = list(my_storageobj.mona.items())
        self.assertEqual(len(elements), 1)
        my_storagedict = MyStorageDictA('second_name')
        last_key = 'some_key'
        last_value = 123

        my_storagedict[last_key] = last_value
        # my_storageobj.mona
        my_storageobj.mona = my_storagedict
        self.assertTrue(last_key in my_storageobj.mona)

        last_items = list(my_storageobj.mona.items())
        self.assertEqual(len(last_items), 1)
        self.assertEqual(my_storagedict[last_key], last_value)

        config.session.execute("DROP TABLE IF EXISTS my_app.first_name")
        config.session.execute("DROP TABLE IF EXISTS my_app.first_name_mona")
        config.session.execute("DROP TABLE IF EXISTS my_app.first_name_mona_0")
        config.session.execute("DROP TABLE IF EXISTS my_app.first_name_mona_1")
        config.session.execute("DROP TABLE IF EXISTS my_app.second_name")

    def test_make_persistent_with_persistent_obj(self):
        o2 = myobj2("obj")
        o2.attr1 = 1
        o2.attr2 = "2"

        d = mydict()
        d[0] = o2
        try:
            d.make_persistent("dict")
        except Exception as ex:
            self.fail("Raised exception unexpectedly.\n" + str(ex))

    def test_int_tuples(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.dictwithtuples")
        d = DictWithTuples("my_app.dictwithtuples")

        what_should_be = dict()
        for i in range(0, 10):
            what_should_be[i] = (i, i + 10)
            d[i] = (i, i + 10)

        user_time.sleep(1)
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

    def test_values_tuples(self):
        # @TypeSpec dict<<key:int>, val0:int, val1:tuple<long,int>, val2:str, val3:tuple<str,float>>
        config.session.execute("DROP TABLE IF EXISTS my_app.dictwithtuples3")
        d = DictWithTuples3("my_app.dictwithtuples3")

        what_should_be = set()
        for i in range(0, 20):
            what_should_be.add((i, (5500000000000000, i + 10), "hola", ("adios", (i + 20.5))))
            d[i] = [i, (5500000000000000, i + 10), "hola", ("adios", (i + 20.5))]

        user_time.sleep(1)
        res = set()
        count = 0
        for item in d.values():
            res.add(tuple(item))
            count += 1

        self.assertEqual(count, len(what_should_be))
        self.assertEqual(what_should_be, res)
        self.assertEqual(what_should_be, res)

    def test_tuples_in_key(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.dictwithtuples2")
        d = DictWithTuples2("my_app.dictwithtuples2")

        for i in range(0, 10):
            d[(i, i), i + 1] = str(i)

        user_time.sleep(1)
        for i in range(0, 10):
            self.assertEqual(d[(i, i), i + 1], str(i))

        self.assertEqual(len(list(d.keys())), 10)

    def test_keys_tuples(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.dictwithtuples2")
        d = DictWithTuples2("my_app.dictwithtuples2")

        what_should_be = set()
        for i in range(0, 10):
            what_should_be.add(((i, i), i + 1))
            d[(i, i), i + 1] = str(i)

        user_time.sleep(1)

        res = set()
        count = 0
        for key in d.keys():
            res.add(tuple(key))
            count += 1

        self.assertEqual(count, len(what_should_be))
        self.assertEqual(what_should_be, res)

    def test_multiple_tuples(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.dictmultipletuples")
        d = DictWithTuples3("my_app.dictmultipletuples")

        what_should_be = dict()
        for i in range(0, 10):
            what_should_be[i] = [i, (5500000000000000, i + 10), "hola", ("adios", (i + 20.5))]
            d[i] = [i, (5500000000000000, i + 10), "hola", ("adios", (i + 20.5))]

        user_time.sleep(2)
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

    def test_int_tuples_null_values(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.dictwithtuples")
        d = DictWithTuples("my_app.dictwithtuples")

        for i in range(0, 10):
            if i % 2 == 0:
                d[i] = (None, i + 10)
            else:
                d[i] = (i, i + 10)

        d = DictWithTuples("my_app.dictwithtuples")
        for i in range(0, 10):
            if i % 2 == 0:
                self.assertEqual(d[i], (None, i + 10))
            else:
                self.assertEqual(d[i], (i, i + 10))

    def test_multi_tuples(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.multituples")
        d = MultiTuples("my_app.multituples")
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

    def test_multiple_tuples_NULL(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.dictmultipletuples")
        d = DictWithTuples3("my_app.dictmultipletuples")

        what_should_be = dict()
        for i in range(0, 10):
            if i % 2 == 0:
                what_should_be[i] = [i, (5500000000000000, i + 10), "hola", ("adios", (i + 20.5))]
                d[i] = [i, (5500000000000000, i + 10), "hola", ("adios", (i + 20.5))]
            else:
                what_should_be[i] = [i, (5500000000000000, None), "hola", (None, (i + 20.5))]
                d[i] = [i, (5500000000000000, None), "hola", (None, (i + 20.5))]

        d = DictWithTuples3("my_app.dictmultipletuples")
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

    def gen_random_date(self):
        return date(year=randint(2000, 2019), month=randint(1, 12), day=randint(1, 28))

    def gen_random_datetime(self):
        return datetime(year=randint(2000, 2019), month=randint(1, 12), day=randint(1, 28),
                        hour=randint(0, 23), minute=randint(0, 59), second=randint(0, 59))

    def gen_random_time(self):
        return time(hour=randint(0, 23), minute=randint(0, 59), second=randint(0, 59))

    def test_multiple_dates(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.dictwithdates")
        d = DictWithDates("my_app.dictwithdates")
        what_should_be = dict()
        for i in range(0, 1):
            keys = self.gen_random_date()
            cols = self.gen_random_date()
            what_should_be[keys] = [cols]
            d[keys] = [cols]

        del d
        import gc
        gc.collect()
        d = DictWithDates("my_app.dictwithdates")

        self.assertEqual(len(list(d.keys())), len(what_should_be.keys()))

        count = 0
        for k in what_should_be.keys():
            count += 1
            self.assertEqual(what_should_be[k], [d[k]])

        self.assertEqual(count, len(list(d)))

    def test_multiple_times(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.dictwithtimes")
        d = DictWithTimes("my_app.dictwithtimes")
        what_should_be = dict()
        for i in range(0, 3):
            keys = self.gen_random_time()
            cols = self.gen_random_time()
            what_should_be[keys] = [cols]
            d[keys] = [cols]

        del d
        import gc
        gc.collect()
        d = DictWithTimes("my_app.dictwithtimes")


        self.assertEqual(len(list(d.keys())), len(what_should_be.keys()))

        count = 0
        for k in what_should_be.keys():
            count += 1
            self.assertEqual(what_should_be[k], [d[k]])

        self.assertEqual(count, len(list(d)))


if __name__ == '__main__':
    unittest.main()
