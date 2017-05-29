import unittest

from hecuba import config
from hecuba.storageobj import StorageObj
from app.words import Words


class StorageObjTest(unittest.TestCase):

    def test_init_empty(self):
        nopars = StorageObj('ksp1.ttta')
        self.assertEqual('ttta', nopars._table)
        self.assertEqual('ksp1', nopars._ksp)

        res = ''
        res1 = config.session.execute(
            'SELECT storage_id, class_name, name, tokens, istorage_props FROM hecuba_app.istorage WHERE storage_id = \'' + str(nopars._storage_id) + '\'')
        for row in res1:
            storage_id, storageobj_classname, name, tokens, istorage_props = row
            res = row
        # [0]
        # storage_id, storageobj_classname, name = res
        self.assertEqual(storage_id, str(nopars._storage_id))
        self.assertEqual(storageobj_classname, 'hecuba.storageobj.StorageObj')
        self.assertEqual(name, 'ksp1.ttta_e419cc665a46313580796ecc755e5342')

        rebuild = StorageObj.build_remotely(res)
        self.assertEqual('ttta', rebuild._table)
        self.assertEqual('ksp1', rebuild._ksp)
        self.assertEqual(storage_id, str(rebuild._storage_id))

        self.assertEqual(nopars._is_persistent, rebuild._is_persistent)
        #self.assertEqual(vars(nopars), vars(rebuild))

    def test_make_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba_app.words_de1b645ce2483509b58cbca4592c1430")
        nopars = Words()
        self.assertFalse(nopars._is_persistent)
        config.batch_size = 1
        config.cache_activated = False
        nopars.ciao = 1
        nopars.ciao2 = "1"
        nopars.ciao3 = [1, 2, 3]
        nopars.ciao4 = (1, 2, 3)
        for i in range(10):
            nopars.words[i] = 'ciao'+str(i)

        count, = config.session.execute("SELECT count(*) FROM system_schema.tables WHERE keyspace_name = 'hecuba_app' and table_name = 'words'")[0]
        self.assertEqual(0, count)

        nopars.make_persistent("hecuba_app.wordsso")

        count, = config.session.execute('SELECT count(*) FROM hecuba_app.words_de1b645ce2483509b58cbca4592c1430')[0]
        self.assertEqual(10, count)

    def test_empty_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba_app.words_de1b645ce2483509b58cbca4592c1430")
        config.session.execute("DROP TABLE IF EXISTS hecuba_app.wordsso_6d5aff9e38263ef3b8b835464659d4ce")
        from app.words import Words
        so = Words()
        so.make_persistent("wordsso")
        so.ciao = "an attribute"
        so.another = 123
        config.batch_size = 1
        config.cache_activated = False
        for i in range(10):
            so.words[i] = str.join(',', map(lambda a: "ciao", range(i)))

        count, = config.session.execute('SELECT count(*) FROM hecuba_app.words_de1b645ce2483509b58cbca4592c1430')[0]
        self.assertEqual(10, count)

        so.delete_persistent()

        count, = config.session.execute('SELECT count(*) FROM hecuba_app.words_de1b645ce2483509b58cbca4592c1430')[0]
        self.assertEqual(0, count)


if __name__ == '__main__':
    unittest.main()
