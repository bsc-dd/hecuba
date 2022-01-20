import unittest

from hecuba import config
from hecuba.hdict import StorageDict
from hecuba.storageobj import StorageObj


class SObj_Basic(StorageObj):
    '''
    @ClassField attr1 int
    @ClassField attr2 double
    @ClassField attr3 str
    '''


class SDict_SimpleTypeSpec(StorageDict):
    '''
    @TypeSpec dict<<id:int>, info:str>
    '''


class SDict_ComplexTypeSpec(StorageDict):
    '''
    @TypeSpec dict<<id:int>, state:tests.withcassandra.storagedict_split_tests.SObj_Basic>
    '''


class SObj_SimpleClassField(StorageObj):
    '''
    @ClassField attr1 int
    @ClassField mydict dict<<key:str>, value:double>
    @ClassField attr3 double
    '''


class SObj_ComplexClassField(StorageObj):
    '''
    @ClassField attr1 int
    @ClassField mydict dict<<key:str>, val:tests.withcassandra.storagedict_split_tests.SObj_Basic>
    @ClassField attr3 double
    '''


class StorageDictSplitTestbase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #config.session.execute("DROP KEYSPACE IF EXISTS my_app", timeout=60)
        config.session.execute(
            "CREATE KEYSPACE IF NOT EXISTS my_app WITH "
            "replication = {'class': 'SimpleStrategy', 'replication_factor': 1};",
            timeout=60)
        pass

    @classmethod
    def tearDownClass(cls):
        #config.session.execute("DROP KEYSPACE IF EXISTS my_app", timeout=60)
        pass

    def test_simple_iterkeys_split(self):
        config.session.execute(
            "CREATE TABLE IF NOT EXISTS my_app.tab30(position int, value text, PRIMARY KEY(position))")
        tablename = "tab30"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        num_inserts = 10000
        what_should_be = set()
        for i in range(num_inserts):
            pd[i] = 'ciao' + str(i)
            what_should_be.add(i)

        count = 0
        res = set()
        for partition in pd.split():
            for val in partition.keys():
                res.add(val)
                count += 1
        self.assertEqual(count, num_inserts)
        self.assertEqual(what_should_be, res)

        count, = config.session.execute('SELECT count(*) FROM my_app.tab30')[0]
        self.assertEqual(count, num_inserts)
        pd.delete_persistent()

    def test_remote_build_iterkeys_split(self):
        config.session.execute(
            "CREATE TABLE IF NOT EXISTS my_app.tab_b0(position int, value text, PRIMARY KEY(position))")
        tablename = "tab_b0"
        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        num_inserts = 10000
        what_should_be = set()
        for i in range(num_inserts):
            pd[i] = 'ciao' + str(i)
            what_should_be.add(i)

        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])
        count = 0
        res = set()
        for partition in pd.split():
            id = partition.storage_id
            from storage.api import getByID
            rebuild = getByID(id)
            for val in rebuild.keys():
                res.add(val)
                count += 1
        self.assertEqual(count, num_inserts)
        self.assertEqual(what_should_be, res)

        count, = config.session.execute('SELECT count(*) FROM my_app.tab_b0')[0]
        self.assertEqual(count, num_inserts)
        pd.delete_persistent()

    def test_composed_iteritems(self):
        config.session.execute(
            "CREATE TABLE IF NOT EXISTS my_app.tab_b1(pid int,time int, value text,x float,y float,z float, PRIMARY KEY(pid,time))")
        tablename = "tab_b1"
        pd = StorageDict(tablename,
                         [('pid', 'int'), ('time', 'int')],
                         [('value', 'text'), ('x', 'double'), ('y', 'double'), ('z', 'double')])
        num_inserts = 10000
        what_should_be = {}
        for i in range(num_inserts):
            pd[i, i + 100] = ['ciao' + str(i), i * 0.1, i * 0.2, i * 0.3]
            what_should_be[i, i + 100] = ('ciao' + str(i), i * 0.1, i * 0.2, i * 0.3)

        count = 0
        res = {}
        for partition in pd.split():
            for key, val in partition.items():
                res[key] = val
                count += 1
        self.assertEqual(count, num_inserts)

        count, = config.session.execute('SELECT count(*) FROM my_app.tab_b1')[0]
        self.assertEqual(count, num_inserts)

        delta = 0.0001
        for i in range(num_inserts):
            a = what_should_be[i, i + 100]
            b = res[i, i + 100]
            self.assertEqual(a[0], b.value)
            self.assertAlmostEquals(a[1], b.x, delta=delta)
            self.assertAlmostEquals(a[2], b.y, delta=delta)
            self.assertAlmostEquals(a[3], b.z, delta=delta)
        pd.delete_persistent()

    def computeItems(self, SDict):
        counter = 0
        for item in SDict.keys():
            counter = counter + 1
        # self.assertEqual(counter, expected)
        return counter

    def test_split_type_spec_basic(self):
        nitems = 1000
        mybook = SDict_SimpleTypeSpec("test_records")
        for id in range(0, nitems):
            mybook[id] = 'someRandomText' + str(id)

        del mybook
        import gc
        gc.collect()
        # verify all data has been written
        myotherbook = SDict_SimpleTypeSpec("test_records")
        self.assertEqual(nitems, self.computeItems(myotherbook))
        # we don't want anything in memory
        del myotherbook

        myfinalbook = SDict_SimpleTypeSpec("test_records")
        # split the dict and assert all the dicts generated contain the expected data
        acc = 0
        nsplits = 0
        for b in myfinalbook.split():  # this split fails
            acc = acc + self.computeItems(b)
            nsplits = nsplits + 1

        self.assertEqual(acc, nitems)
        myfinalbook.delete_persistent()

    def test_split_type_spec_complex(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.SObj_ComplexClassField")
        nitems = 10
        mybook = SDict_ComplexTypeSpec("experimentx")
        for id in range(0, nitems):
            mybook[id] = SObj_Basic()
            mybook[id].attr1 = id
            mybook[id].attr2 = id / nitems
            mybook[id].attr3 = "basicobj" + str(id)

        del mybook

        # verify all data has been written
        myotherbook = SDict_ComplexTypeSpec("experimentx")
        self.assertEqual(nitems, self.computeItems(myotherbook))
        # we don't want anything in memory
        del myotherbook

        myfinalbook = SDict_ComplexTypeSpec("experimentx")
        # split the dict and assert all the dicts generated contain the expected data
        acc = 0
        nsplits = 0
        for b in myfinalbook.split():  # this split fails
            acc = acc + self.computeItems(b)
            nsplits = nsplits + 1

        self.assertEqual(acc, nitems)
        myfinalbook.delete_persistent()

    def test_split_class_field_simple(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.SObj_SimpleClassField")
        nitems = 80
        mybook = SObj_SimpleClassField("so_split_dict_simple")
        mybook.attr1 = nitems
        mybook.attr3 = nitems / 100
        for id in range(0, nitems):
            key_text = 'so_split_dict_simple' + str(id)
            mybook.mydict[key_text] = id / nitems

        del mybook

        # verify all data has been written
        myotherbook = SObj_SimpleClassField("so_split_dict_simple")
        self.assertEqual(nitems, self.computeItems(myotherbook.mydict))
        # we don't want anything in memory
        del myotherbook

        myfinalbook = SObj_SimpleClassField("so_split_dict_simple")
        # split the dict and assert all the dicts generated contain the expected data
        acc = 0
        nsplits = 0
        for b in myfinalbook.mydict.split():  # this split fails
            acc = acc + self.computeItems(b)
            nsplits = nsplits + 1

        self.assertEqual(acc, nitems)
        myfinalbook.delete_persistent()

    def test_split_class_field_complex(self):
        nitems = 50
        mybook = SObj_ComplexClassField("so_split_dict_complex")
        mybook.attr1 = nitems
        mybook.attr3 = nitems / 100
        for id in range(0, nitems):
            key_text = 'so_split_dict_simple' + str(id)
            so = SObj_Basic()
            so.attr1 = id
            so.attr2 = id / nitems
            so.attr3 = 'someInnerRandomText' + str(id)
            mybook.mydict[key_text] = so

        del mybook

        # verify all data has been written
        myotherbook = SObj_ComplexClassField("so_split_dict_complex")
        self.assertEqual(nitems, self.computeItems(myotherbook.mydict))
        # we don't want anything in memory
        del myotherbook

        myfinalbook = SObj_ComplexClassField("so_split_dict_complex")
        # split the dict and assert all the dicts generated contain the expected data
        acc = 0
        nsplits = 0
        for b in myfinalbook.mydict.split():  # this split fails
            acc = acc + self.computeItems(b)
            nsplits = nsplits + 1

        self.assertEqual(acc, nitems)
        myfinalbook.delete_persistent()

    def test_len_on_split(self):
        ninserts = 100
        obj = SDict_SimpleTypeSpec("test_split_len")
        for i in range(ninserts):
            obj[i] = str(f"test_split_len{i}")
        nin = len(obj)

        count = 0
        for chunk in obj.split():
            count = count + len(chunk)

        self.assertEqual(count, ninserts)
        obj.delete_persistent()

    '''
    def test_remote_build_composed_iteritems(self):
         config.session.execute(
            "CREATE TABLE IF NOT EXISTS my_app.tab_b2(pid int,time int, value text,x float,y float,z float, PRIMARY KEY(pid,time))")
        tablename = "tab_b2"
        pd = StorageDict(tablename,
                         [('pid', 'int'), ('time', 'int')],
                         [('value', 'text'), ('x', 'float'), ('y', 'float'), ('z', 'float')])

        what_should_be = {}
        for i in range(10000):
            pd[i, i + 100] = ('ciao' + str(i), i * 0.1, i * 0.2, i * 0.3)
            what_should_be[i, i + 100] = ('ciao' + str(i), i * 0.1, i * 0.2, i * 0.3)

        del pd

        count, = config.session.execute('SELECT count(*) FROM my_app.tab_b2')[0]
        self.assertEqual(count, 10000)
        pd = StorageDict(tablename,
                         [('pid', 'int'), ('time', 'int')],
                         [('value', 'text'), ('x', 'float'), ('y', 'float'), ('z', 'float')])
        count = 0
        res = {}
        for partition in pd.split():
            id = partition.storage_id
            from storage.api import getByID
            rebuild = getByID(id)
            for key, val in rebuild.items():
                res[key] = val
                count += 1
        self.assertEqual(count, 10000)
        delta = 0.0001
        for i in range(10000):
            a = what_should_be[i, i + 100]
            b = res[i, i + 100]
            self.assertEqual(a[0], b.value)
            self.assertAlmostEquals(a[1], b.x, delta=delta)
            self.assertAlmostEquals(a[2], b.y, delta=delta)
            self.assertAlmostEquals(a[3], b.z, delta=delta)
    '''


class StorageDictSlitTestVnodes(StorageDictSplitTestbase):
    @classmethod
    def setUpClass(cls):
        from hfetch import disconnectCassandra
        disconnectCassandra()
        from .. import test_config, set_ccm_cluster
        test_config.ccm_cluster.clear()
        set_ccm_cluster()
        from .. import TEST_DEBUG
        try:
            test_config.ccm_cluster.populate(3, use_vnodes=True).start()
        except Exception as ex:
            if not TEST_DEBUG:
                raise ex

        import hfetch
        import hecuba
        import importlib
        importlib.reload(hfetch)
        import importlib
        importlib.reload(hecuba)
        config.session.execute("DROP KEYSPACE IF EXISTS my_app")
        config.session.execute(
            "CREATE KEYSPACE IF NOT EXISTS my_app WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")
        super(StorageDictSplitTestbase, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        config.session.execute("DROP KEYSPACE IF EXISTS my_app")
        from .. import test_config
        from hfetch import disconnectCassandra
        disconnectCassandra()

        test_config.ccm_cluster.clear()
        from .. import set_up_default_cassandra
        set_up_default_cassandra()


if __name__ == '__main__':
    unittest.main()
