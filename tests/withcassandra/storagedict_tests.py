import unittest

from hecuba import config
from hecuba.hdict import StorageDict
from app.words import Words
import uuid


class StorageObjTest(unittest.TestCase):

    def test_init_empty(self):
        # done
        tablename = "tab1"
        tokens = [(1l, 2l), (2l, 3l), (3l, 4l)]
        nopars = StorageDict([('position', 'int')], [('value', 'int')], tablename, tokens)
        self.assertEqual("tab1", nopars._table)
        self.assertEqual(config.execution_name, nopars._ksp)

        res = ''
        res1 = config.session.execute(
            'SELECT storage_id, primary_keys, columns, class_name, name, tokens, istorage_props FROM hecuba.istorage WHERE storage_id = ' + str(nopars._storage_id) + '')
        for row in res1:
            res = row

        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.' + tablename), nopars._storage_id)
        self.assertEqual(nopars.__class__.__module__, 'hecuba.hdict')
        self.assertEqual(nopars.__class__.__name__, 'StorageDict')

        rebuild = StorageDict.build_remotely(res)
        self.assertEqual('tab1', rebuild._table)
        self.assertEqual(config.execution_name, rebuild._ksp)
        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.' + tablename), rebuild._storage_id)

        self.assertEqual(nopars._is_persistent, rebuild._is_persistent)

    def test_flush_items_cached(self):
        # in process
        config.session.execute("CREATE TABLE IF NOT EXISTS hecuba.tab1(pk1 int, val1 text, PRIMARY KEY(pk1))")
        config.cache_activated = True
        tablename = "tab1"
        tokens = [(1l, 2l), (2l, 3l), (3l, 4l)]
        pd = StorageDict([('position', 'int')], [('value', 'int')], tablename, tokens)
        config.batch_size = 101
        for i in range(100):
            pd[int(i)] = 'ciao' + str(i)
        count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
        self.assertEqual(count, 0)
        pd._flush_items()
        count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
        self.assertEqual(count, 100)

    def test_make_persistent(self):
        # done
        config.session.execute("DROP TABLE IF EXISTS hecuba.text_7ac343c2eeb1360caae83c606d5da25c")
        nopars = Words()
        self.assertFalse(nopars._is_persistent)
        config.batch_size = 1
        config.cache_activated = False
        nopars.ciao = 1
        nopars.ciao2 = "1"
        nopars.ciao3 = [1, 2, 3]
        nopars.ciao4 = (1, 2, 3)
        for i in range(10):
            nopars.text[i] = 'ciao'+str(i)

        count, = config.session.execute("SELECT count(*) FROM system_schema.tables WHERE keyspace_name = 'hecuba' and table_name = 'text_7ac343c2eeb1360caae83c606d5da25c'")[0]
        self.assertEqual(0, count)

        nopars.make_persistent("wordsso")

        count, = config.session.execute('SELECT count(*) FROM hecuba.text_7ac343c2eeb1360caae83c606d5da25c')[0]
        self.assertEqual(10, count)

    def test_empty_persistent(self):
        # done
        config.session.execute("DROP TABLE IF EXISTS hecuba.wordsso_6d439fbe0b0334779ebc36da44f2e3b7")
        config.session.execute("DROP TABLE IF EXISTS hecuba.text_7ac343c2eeb1360caae83c606d5da25c")
        from app.words import Words
        so = Words()
        so.make_persistent("wordsso")
        so.ciao = "an attribute"
        so.another = 123
        config.batch_size = 1
        config.cache_activated = False
        for i in range(10):
            so.text[i] = str.join(',', map(lambda a: "ciao", range(i)))

        count, = config.session.execute('SELECT count(*) FROM hecuba.text_7ac343c2eeb1360caae83c606d5da25c')[0]
        self.assertEqual(10, count)

        so.delete_persistent()
        so.text.delete_persistent()

        count, = config.session.execute('SELECT count(*) FROM hecuba.text_7ac343c2eeb1360caae83c606d5da25c')[0]
        self.assertEqual(0, count)


if __name__ == '__main__':
    unittest.main()
