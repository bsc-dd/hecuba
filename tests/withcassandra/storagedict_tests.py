import unittest

from hecuba import config
from hecuba.hdict import StorageDict
from app.words import Words
import uuid
import time


class StorageDictTest(unittest.TestCase):
    def test_init_empty(self):
        config.session.execute("DROP TABLE IF EXISTS ksp.tab1")
        tablename = "ksp.tab1"
        tokens = [(1l, 2l), (2l, 3l), (3l, 4l)]
        nopars = StorageDict([('position', 'int')], [('value', 'int')], tablename, tokens)
        self.assertEqual("tab1", nopars._table)
        self.assertEqual("ksp", nopars._ksp)

        res = config.session.execute(
            'SELECT storage_id, primary_keys, columns, class_name, name, tokens, istorage_props ' +
            'FROM hecuba.istorage WHERE storage_id = %s', [nopars._storage_id])[0]

        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, tablename), nopars._storage_id)
        self.assertEqual(nopars.__class__.__module__, 'hecuba.hdict')
        self.assertEqual(nopars.__class__.__name__, 'StorageDict')

        rebuild = StorageDict.build_remotely(res)
        self.assertEqual('tab1', rebuild._table)
        self.assertEqual("ksp", rebuild._ksp)
        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, tablename), rebuild._storage_id)

        self.assertEqual(nopars._is_persistent, rebuild._is_persistent)

    def test_init_empty_def_keyspace(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba.tab1")
        tablename = "tab1"
        tokens = [(1l, 2l), (2l, 3l), (3l, 4l)]
        nopars = StorageDict([('position', 'int')], [('value', 'int')], tablename, tokens)
        self.assertEqual("tab1", nopars._table)
        self.assertEqual(config.execution_name, nopars._ksp)

        res = config.session.execute(
            'SELECT storage_id, primary_keys, columns, class_name, name, tokens, istorage_props ' +
            'FROM hecuba.istorage WHERE storage_id = %s', [nopars._storage_id])[0]

        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.' + tablename), nopars._storage_id)
        self.assertEqual(nopars.__class__.__module__, 'hecuba.hdict')
        self.assertEqual(nopars.__class__.__name__, 'StorageDict')

        rebuild = StorageDict.build_remotely(res)
        self.assertEqual('tab1', rebuild._table)
        self.assertEqual(config.execution_name, rebuild._ksp)
        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.' + tablename), rebuild._storage_id)

        self.assertEqual(nopars._is_persistent, rebuild._is_persistent)

    def test_simple_insertions(self):
        # in process
        config.session.execute(
            "CREATE TABLE IF NOT EXISTS hecuba.tab10(position int, value text, PRIMARY KEY(position))")
        tablename = "tab10"
        tokens = [(1l, 2l), (2l, 3l), (3l, 4l)]
        pd = StorageDict([('position', 'int')], [('value', 'text')], tablename, tokens)

        for i in range(100):
            pd[i] = 'ciao' + str(i)
        time.sleep(4)
        count, = config.session.execute('SELECT count(*) FROM hecuba.tab10')[0]
        del pd
        self.assertEqual(count, 100)

    def test_make_persistent(self):
        # done
        config.session.execute("DROP TABLE IF EXISTS hecuba.t_make_words")
        nopars = Words()
        self.assertFalse(nopars._is_persistent)
        nopars.ciao = 1
        nopars.ciao2 = "1"
        nopars.ciao3 = [1, 2, 3]
        nopars.ciao4 = (1, 2, 3)
        for i in range(10):
            nopars.words[i] = 'ciao' + str(i)

        count, = config.session.execute(
            "SELECT count(*) FROM system_schema.tables WHERE keyspace_name = 'hecuba' and table_name = 't_make_words'")[
            0]
        self.assertEqual(0, count)

        nopars.make_persistent("t_make")

        del nopars
        count, = config.session.execute('SELECT count(*) FROM hecuba.t_make_words')[0]
        self.assertEqual(10, count)

    def test_empty_persistent(self):
        # done
        config.session.execute("DROP TABLE IF EXISTS hecuba.wordsso_words")
        config.session.execute("DROP TABLE IF EXISTS hecuba.wordsso")
        from app.words import Words
        so = Words()
        so.make_persistent("wordsso")
        so.ciao = "an attribute"
        so.another = 123
        config.batch_size = 1
        config.cache_activated = False
        for i in range(10):
            so.words[i] = str.join(',', map(lambda a: "ciao", range(i)))

        del so
        count, = config.session.execute('SELECT count(*) FROM hecuba.wordsso_words')[0]
        self.assertEqual(10, count)

        so = Words("wordsso")
        so.delete_persistent()
        so.words.delete_persistent()

        count, = config.session.execute('SELECT count(*) FROM hecuba.wordsso_words')[0]
        self.assertEqual(0, count)

    def test_simple_iteritems_test(self):
        # in process
        config.session.execute("DROP TABLE IF EXISTS hecuba.tab_a1")
        config.session \
            .execute("CREATE TABLE IF NOT EXISTS hecuba.tab_a1(position int, value text, PRIMARY KEY(position))")

        pd = StorageDict([('position', 'int')], [('value', 'text')], "tab_a1")

        what_should_be = {}
        for i in range(100):
            pd[i] = 'ciao' + str(i)
            what_should_be[i] = 'ciao' + str(i)
        del pd
        count, = config.session.execute('SELECT count(*) FROM hecuba.tab_a1')[0]
        self.assertEqual(count, 100)
        pd = StorageDict([('position', 'int')], [('value', 'text')], "tab_a1")
        count = 0
        res = {}
        for key, val in pd.iteritems():
            res[key] = val
            count += 1
        self.assertEqual(count, 100)
        self.assertEqual(what_should_be, res)

    def test_simple_itervalues_test(self):
        # in process
        config.session.execute("DROP TABLE IF EXISTS hecuba.tab_a2")
        config.session.execute(
            "CREATE TABLE IF NOT EXISTS hecuba.tab_a2(position int, value text, PRIMARY KEY(position))")
        tablename = "tab_a2"
        pd = StorageDict([('position', 'int')], [('value', 'text')], tablename)

        what_should_be = set()
        for i in range(100):
            pd[i] = 'ciao' + str(i)
            what_should_be.add('ciao' + str(i))
        del pd
        count, = config.session.execute('SELECT count(*) FROM hecuba.tab_a2')[0]

        self.assertEqual(count, 100)

        pd = StorageDict([('position', 'int')], [('value', 'text')], tablename)
        count = 0
        res = set()
        for val in pd.itervalues():
            res.add(val)
            count += 1
        self.assertEqual(count, 100)
        self.assertEqual(what_should_be, res)

    def test_simple_iterkeys_test(self):
        # in process
        config.session.execute("DROP TABLE IF EXISTS hecuba.tab_a3")
        config.session.execute(
            "CREATE TABLE IF NOT EXISTS hecuba.tab_a3(position int, value text, PRIMARY KEY(position))")
        tablename = "tab_a3"
        pd = StorageDict([('position', 'int')], [('value', 'text')], tablename)

        what_should_be = set()
        for i in range(100):
            pd[i] = 'ciao' + str(i)
            what_should_be.add(i)
        del pd
        count, = config.session.execute('SELECT count(*) FROM hecuba.tab_a3')[0]
        self.assertEqual(count, 100)
        pd = StorageDict([('position', 'int')], [('value', 'text')], tablename)
        count = 0
        res = set()
        for val in pd.iterkeys():
            res.add(val)
            count += 1
        self.assertEqual(count, 100)
        self.assertEqual(what_should_be, res)

    def test_simple_contains(self):
        # in process
        config.session.execute("DROP TABLE IF EXISTS hecuba.tab_a4")
        config.session.execute(
            "CREATE TABLE hecuba.tab_a4(position int, value text, PRIMARY KEY(position))")
        tablename = "tab_a4"
        pd = StorageDict([('position', 'int')], [('value', 'text')], tablename)

        for i in range(100):
            pd[i] = 'ciao' + str(i)
        del pd
        count, = config.session.execute('SELECT count(*) FROM hecuba.tab_a4')[0]
        self.assertEqual(count, 100)

        pd = StorageDict([('position', 'int')], [('value', 'text')], tablename)
        for i in range(100):
            self.assertTrue(i in pd)

    def test_composed_iteritems_test(self):
        # in process
        config.session.execute("DROP TABLE IF EXISTS hecuba.tab12")
        config.session.execute(
            "CREATE TABLE IF NOT EXISTS hecuba.tab12(pid int,time int, value text,x float,y float,z float, PRIMARY KEY(pid,time))")
        tablename = "tab12"
        pd = StorageDict([('pid', 'int'), ('time', 'int')],
                         [('value', 'text'),
                          ('x', 'float'),
                          ('y', 'float'), ('z', 'float')], tablename)

        what_should_be = {}
        for i in range(100):
            pd[i, i + 100] = ('ciao' + str(i), i * 0.1, i * 0.2, i * 0.3)
            what_should_be[i, i + 100] = ('ciao' + str(i), i * 0.1, i * 0.2, i * 0.3)

        del pd

        count, = config.session.execute('SELECT count(*) FROM hecuba.tab12')[0]
        self.assertEqual(count, 100)
        pd = StorageDict([('pid', 'int'), ('time', 'int')],
                         [('value', 'text'),
                          ('x', 'float'),
                          ('y', 'float'), ('z', 'float')], tablename)
        count = 0
        res = {}
        for key, val in pd.iteritems():
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


if __name__ == '__main__':
    unittest.main()
