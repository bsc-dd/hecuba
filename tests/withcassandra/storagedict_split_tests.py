import unittest

from hecuba import config
from hecuba.hdict import StorageDict


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
                         [('value', 'text'), ('x', 'float'), ('y', 'float'), ('z', 'float')])
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
