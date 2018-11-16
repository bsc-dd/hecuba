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
    @TypeSpec <<id:int>,info:str>
    '''


class SDict_ComplexTypeSpec(StorageDict):
    '''
    @TypeSpec <<id:int>,state:tests.withcassandra.storagedict_split_tests.SObj_Basic>
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
    @ClassField mydict dict <<key:str>,state:tests.withcassandra.storagedict_split_tests.SObj_Basic>
    @ClassField attr3 double
    '''


class StorageDictSplitTest(unittest.TestCase):

    def test_simple_iterkeys_split_test(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab30")
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
        del pd
        count, = config.session.execute('SELECT count(*) FROM my_app.tab30')[0]
        self.assertEqual(count, num_inserts)

        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])

        count = 0
        res = set()
        for partition in pd.split():
            for val in partition.iterkeys():
                res.add(val)
                count += 1
        self.assertEqual(count, num_inserts)
        self.assertEqual(what_should_be, res)

    def test_remote_build_iterkeys_split_test(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab_b0")
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
        del pd
        count, = config.session.execute('SELECT count(*) FROM my_app.tab_b0')[0]
        self.assertEqual(count, num_inserts)

        pd = StorageDict(tablename,
                         [('position', 'int')],
                         [('value', 'text')])

        count = 0
        res = set()
        for partition in pd.split():
            id = partition.getID()
            from storage.api import getByID
            rebuild = getByID(id)
            for val in rebuild.iterkeys():
                res.add(val)
                count += 1
        self.assertEqual(count, num_inserts)
        self.assertEqual(what_should_be, res)

    def test_composed_iteritems_test(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab_b1")
        config.session.execute(
            "CREATE TABLE IF NOT EXISTS my_app.tab_b1(pid int,time int, value text,x float,y float,z float, PRIMARY KEY(pid,time))")
        tablename = "tab_b1"
        pd = StorageDict(tablename,
                         [('pid', 'int'), ('time', 'int')],
                         [('value', 'text'), ('x', 'double'), ('y', 'double'), ('z', 'double')])
        num_inserts = 10000
        what_should_be = {}
        for i in range(num_inserts):
            pd[i, i + 100] = ('ciao' + str(i), i * 0.1, i * 0.2, i * 0.3)
            what_should_be[i, i + 100] = ('ciao' + str(i), i * 0.1, i * 0.2, i * 0.3)

        del pd

        count, = config.session.execute('SELECT count(*) FROM my_app.tab_b1')[0]
        self.assertEqual(count, num_inserts)
        pd = StorageDict(tablename,
                         [('pid', 'int'), ('time', 'int')],
                         [('value', 'text'), ('x', 'float'), ('y', 'float'), ('z', 'float')])
        count = 0
        res = {}
        for partition in pd.split():
            for key, val in partition.iteritems():
                res[key] = val
                count += 1
        self.assertEqual(count, num_inserts)
        delta = 0.0001
        for i in range(num_inserts):
            a = what_should_be[i, i + 100]
            b = res[i, i + 100]
            self.assertEqual(a[0], b.value)
            self.assertAlmostEquals(a[1], b.x, delta=delta)
            self.assertAlmostEquals(a[2], b.y, delta=delta)
            self.assertAlmostEquals(a[3], b.z, delta=delta)

    def computeItems(self, SDict):
        expected = len(SDict)
        counter = 0
        for item in SDict.iterkeys():
            counter = counter + 1
        # self.assertEqual(counter, expected)
        return counter

    # def testSplitTypeSpecBasic(self):
    #     config.session.execute("DROP TABLE IF EXISTS my_app.test_records")
    #     nitems = 1000
    #     mybook = SDict_SimpleTypeSpec("test_records")
    #     for id in xrange(0, nitems):
    #         mybook[id] = 'someRandomText' + str(id)
    #
    #     del mybook
    #
    #     # verify all data has been written
    #     myotherbook = SDict_SimpleTypeSpec("test_records")
    #     self.assertEqual(nitems, self.computeItems(myotherbook))
    #     # we don't want anything in memory
    #     del myotherbook
    #
    #     myfinalbook = SDict_SimpleTypeSpec("test_records")
    #     # split the dict and assert all the dicts generated contain the expected data
    #     acc = 0
    #     nsplits = 0
    #     for b in myfinalbook.split():  # this split fails
    #         acc = acc + self.computeItems(b)
    #         nsplits = nsplits + 1
    #
    #     self.assertEqual(acc, nitems)

    # def testSplitTypeSpecComplex(self):
    #     config.session.execute("DROP TABLE IF EXISTS my_app.experimentx")
    #     nitems = 10
    #     mybook = SDict_ComplexTypeSpec("experimentx")
    #     for id in xrange(0, nitems):
    #         mybook[id] = SObj_Basic()
    #         mybook[id].attr1 = id
    #         mybook[id].attr2 = id / nitems
    #         mybook[id].attr3 = "basicobj" + str(id)
    #
    #     del mybook
    #
    #     # verify all data has been written
    #     myotherbook = SDict_ComplexTypeSpec("experimentx")
    #     self.assertEqual(nitems, self.computeItems(myotherbook))
    #     # we don't want anything in memory
    #     del myotherbook
    #
    #     myfinalbook = SDict_ComplexTypeSpec("experimentx")
    #     # split the dict and assert all the dicts generated contain the expected data
    #     acc = 0
    #     nsplits = 0
    #     for b in myfinalbook.split():  # this split fails
    #         acc = acc + self.computeItems(b)
    #         nsplits = nsplits + 1
    #
    #     self.assertEqual(acc, nitems)

    def testSplitClassFieldSimple(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.so_split_dict_simple")
        nitems = 80
        mybook = SObj_SimpleClassField("so_split_dict_simple")
        mybook.attr1 = nitems
        mybook.attr3 = nitems / 100
        for id in xrange(0, nitems):
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

    # def testSplitClassFieldComplex(self):
    #     config.session.execute("DROP TABLE IF EXISTS my_app.so_split_dict_complex")
    #     nitems = 250
    #     mybook = SObj_ComplexClassField("so_split_dict_complex")
    #     mybook.attr1 = nitems
    #     mybook.attr3 = nitems / 100
    #     for id in xrange(0, nitems):
    #         key_text = 'so_split_dict_simple' + str(id)
    #         so = SObj_Basic()
    #         so.attr1 = id
    #         so.attr2 = id / nitems
    #         so.attr3 = 'someInnerRandomText' + str(id)
    #         mybook.mydict[key_text] = so
    #
    #     del mybook
    #
    #     # verify all data has been written
    #     myotherbook = SObj_ComplexClassField("so_split_dict_complex")
    #     self.assertEqual(nitems, self.computeItems(myotherbook.mydict))
    #     # we don't want anything in memory
    #     del myotherbook
    #
    #     myfinalbook = SObj_ComplexClassField("so_split_dict_complex")
    #     # split the dict and assert all the dicts generated contain the expected data
    #     acc = 0
    #     nsplits = 0
    #     for b in myfinalbook.mydict.split():  # this split fails
    #         acc = acc + self.computeItems(b)
    #         nsplits = nsplits + 1
    #
    #     self.assertEqual(acc, nitems)

    '''
    def test_remote_build_composed_iteritems_test(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.tab_b2")
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
            id = partition.getID()
            from storage.api import getByID
            rebuild = getByID(id)
            for key, val in rebuild.iteritems():
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


if __name__ == '__main__':
    unittest.main()
