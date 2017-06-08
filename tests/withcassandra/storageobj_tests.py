import unittest
import uuid

from hecuba.IStorage import IStorage
from app.words import Words
from hecuba import config
from hecuba.storageobj import StorageObj

class Result(StorageObj):
    '''
    @ClassField instances dict<<word:str>,instances:int>
    '''
    pass



class TestStorageObj(StorageObj):
    '''
       @ClassField test dict<<position:int>,text:str>
    '''
    pass


class TestStorageIndexedArgsObj(StorageObj):
    '''
       @ClassField test dict<<position:int>,x:float,y:float,z:float>
       @Index_on test x,y,z
    '''
    pass


class Test2StorageObj(StorageObj):
    '''
       @ClassField name str
       @ClassField age int

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
        name, tkns = \
        config.session.execute("SELECT name,tokens FROM hecuba.istorage WHERE storage_id = %s", [nopars._storage_id])[
            0]

        self.assertEqual(name, config.execution_name + '.tt1')
        self.assertEqual(tkns, r.tokens)

    def test_init_create_pdict(self):

        class res: pass

        r = res()
        r.ksp = config.execution_name
        r.name = u'tt1'
        r.class_name = r.class_name = str(TestStorageObj.__module__) + "." + TestStorageObj.__name__
        r.storage_id = uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.tt1')
        r.tokens = IStorage._discrete_token_ranges(
            [8508619251581300691, 8514581128764531689, 8577968535836399533, 8596162846302799189,
             8603491526474728284, 8628291680139169981, 8687301163739303017, 9111581078517061776])
        r.istorage_props = {}
        nopars = StorageObj.build_remotely(r)
        self.assertEqual('tt1', nopars._table)
        self.assertEqual(config.execution_name, nopars._ksp)
        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.tt1'), nopars._storage_id)
        name, tkns = \
            config.session.execute("SELECT name,tokens FROM hecuba.istorage WHERE storage_id = %s",
                                   [nopars._storage_id])[0]
        self.assertEqual(name, config.execution_name + '.tt1')
        self.assertEqual(tkns, r.tokens)

        tkns = IStorage._discrete_token_ranges(
            [8508619251581300691, 8514581128764531689, 8577968535836399533, 8596162846302799189,
             8603491526474728284, 8628291680139169981, 8687301163739303017, 9111581078517061776])
        nopars = Result(name='tt1',
                        tokens=tkns)
        self.assertEqual('tt1', nopars._table)
        self.assertEqual(config.execution_name, nopars._ksp)
        self.assertEqual(uuid.uuid3(uuid.NAMESPACE_DNS, config.execution_name + '.tt1'), nopars._storage_id)
        self.assertEqual(True, nopars._is_persistent)
        self.assertTrue(hasattr(nopars, 'instances'))
        name, read_tkns = config.session.execute("SELECT name,tokens FROM hecuba.istorage WHERE storage_id = %s",
                                                 [nopars._storage_id])[0]

        self.assertEqual(name, config.execution_name + '.tt1')
        self.assertEqual(tkns, read_tkns)

    def test_init_empty(self):
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

        rebuild = StorageObj.build_remotely(res)
        self.assertEqual('ttta', rebuild._table)
        self.assertEqual('ksp1', rebuild._ksp)
        self.assertEqual(storage_id, rebuild._storage_id)

        self.assertEqual(nopars._is_persistent, rebuild._is_persistent)
        # self.assertEqual(vars(nopars), vars(rebuild))

    def test_make_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba_test.words")
        nopars = Words()
        self.assertFalse(nopars._is_persistent)
        nopars.ciao = 1
        nopars.ciao2 = "1"
        nopars.ciao3 = [1, 2, 3]
        nopars.ciao4 = (1, 2, 3)
        for i in range(10):
            nopars.words[i] = 'ciao' + str(i)

        count, = config.session.execute(
            "SELECT count(*) FROM system_schema.tables WHERE keyspace_name = 'hecuba_test' and table_name = 'words'")[0]
        self.assertEqual(0, count)

        nopars.make_persistent("hecuba_test.wordsso")

        count, = config.session.execute('SELECT count(*) FROM hecuba_test.wordsso_words')[0]
        self.assertEqual(10, count)

    def test_empty_persistent(self):
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
        so = Words()
        so.make_persistent("wordsso")
        so.delete_persistent()

        count, = config.session.execute('SELECT count(*) FROM hecuba.wordsso_words')[0]
        self.assertEqual(0, count)

    def test_simple_stores_after_make_persistent(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba.t2")
        so = Test2StorageObj()
        so.name = 'caio'
        so.age = 1000
        so.make_persistent("t2")
        count, = config.session.execute("SELECT COUNT(*) FROM hecuba.t2")[0]
        self.assertEqual(count, 1)
        self.assertEqual(so.name, 'caio')
        self.assertEqual(so.age, 1000)

    def test_simple_attributes(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba.t2")
        so = Test2StorageObj()
        so.make_persistent("t2")
        so.name = 'caio'
        so.age = 1000
        count, = config.session.execute("SELECT COUNT(*) FROM hecuba.t2")[0]
        self.assertEqual(count, 1)
        self.assertEqual(so.name, 'caio')
        self.assertEqual(so.age, 1000)

    def test_parse_index_on(self):
        a = TestStorageIndexedArgsObj()
        self.assertEqual(a.test._indexed_args, ['x', 'y', 'z'])
        a.make_persistent('tparse.t1')
        from storage.api import getByID
        b = getByID(a.getID())
        self.assertEqual(b.test._indexed_args, ['x', 'y', 'z'])





if __name__ == '__main__':
    unittest.main()
