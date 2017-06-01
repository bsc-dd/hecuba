import unittest
import uuid

from IStorage import IStorage
from hdict import StorageDict
from hecuba import config
from hecuba.storageobj import StorageObj
from app.words import Words
from result import Result


class TestStorageObj(StorageObj):
    '''
       @ClassField test dict<<position:int>,text:str>
    '''
    pass


class StorageObjTest(unittest.TestCase):
    def test_build_remotely(self):

        class res: pass

        r = res()
        r.ksp = config.execution_name
        r.name = 'tt1'
        r.class_name = str(TestStorageObj.__module__) + "." + TestStorageObj.__name__
        r.tokens = IStorage._discrete_token_ranges(
            [8508619251581300691, 8514581128764531689, 8577968535836399533, 8596162846302799189,
             8603491526474728284, 8628291680139169981, 8687301163739303017, 9111581078517061776])
        r.storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.tt1')
        r.istorage_props = {}
        nopars = StorageObj.build_remotely(r)
        self.assertEqual('tt1', nopars._table)
        self.assertEqual(config.execution_name, nopars._ksp)
        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.tt1'), nopars._storage_id)
        config.session.execute.assert_called()

    def test_init_create_pdict(self):

        class res: pass

        r = res()
        r.ksp = config.execution_name
        r.name = u'tt1'
        r.class_name = r.class_name = str(TestStorageObj.__module__) + "." + TestStorageObj.__name__
        r.storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.tt1')
        r.tokens = IStorage._discrete_token_ranges([8508619251581300691, 8514581128764531689, 8577968535836399533, 8596162846302799189,
                  8603491526474728284, 8628291680139169981, 8687301163739303017, 9111581078517061776])
        r.istorage_props = {}
        nopars = StorageObj.build_remotely(r)
        self.assertEqual('tt1', nopars._table)
        self.assertEqual(config.execution_name, nopars._ksp)
        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.tt1'), nopars._storage_id)
        config.session.execute.assert_called()

        config.session.execute = Mock(return_value=None)
        nopars = Result(name='tt1',
                        storage_id='ciao',
                        tokens=[8508619251581300691, 8514581128764531689, 8577968535836399533, 8596162846302799189,
                                8603491526474728284, 8628291680139169981, 8687301163739303017, 9111581078517061776])
        self.assertEqual('tt1', nopars._table)
        self.assertEqual(config.execution_name, nopars._ksp)
        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.tt1'), nopars._storage_id)
        self.assertEqual(True, nopars._is_persistent)
        self.assertTrue(hasattr(nopars, 'instances'))
        self.assertIsInstance(nopars.instances, StorageDict)
        config.session.execute.assert_not_called()

    def test_init_empty(self):
        nopars = StorageObj('ksp1.ttta')
        self.assertEqual('ttta', nopars._table)
        self.assertEqual('ksp1', nopars._ksp)

        res = ''
        res1 = config.session.execute(
            'SELECT storage_id, class_name, name, tokens, istorage_props FROM hecuba_app.istorage WHERE storage_id = \'' + str(
                nopars._storage_id) + '\'')
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
        # self.assertEqual(vars(nopars), vars(rebuild))

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
            nopars.words[i] = 'ciao' + str(i)

        count, = config.session.execute(
            "SELECT count(*) FROM system_schema.tables WHERE keyspace_name = 'hecuba_app' and table_name = 'words'")[0]
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
