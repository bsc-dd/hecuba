import unittest

from hecuba.dict import PersistentDict
from mock import Mock, call, MagicMock
from hecuba import session, config

from hecuba.storageobj import StorageObj
from words import Words


class StorageObjTest(unittest.TestCase):

    def test_init_empty(self):
        nopars = StorageObj('ksp1.ttta')
        self.assertEqual('ttta', nopars._table)
        self.assertEqual('ksp1', nopars._ksp)
        try:
            import uuid
            uuid.UUID(nopars._myuuid)
        except:
            self.fail('bad format myuuid')


        res=session.execute(
            'SELECT  blockid, storageobj_classname, ksp, tab, obj_type FROM hecuba.blocks WHERE blockid = %s', [nopars._myuuid])[0]
        blockid, storageobj_classname, ksp, tab, obj_type = res
        self.assertEqual(blockid, nopars._myuuid)
        self.assertEqual(storageobj_classname, 'hecuba.storageobj.StorageObj')
        self.assertEqual(ksp, nopars._ksp)
        self.assertEqual(tab, nopars._table)
        self.assertEqual(obj_type, 'hecuba')

        rebuild = StorageObj.build_remotely(res)
        self.assertEqual('ttta', rebuild._table)
        self.assertEqual('ksp1', rebuild._ksp)
        self.assertEqual(blockid, rebuild._myuuid)

        self.assertEqual(nopars._persistent, rebuild._persistent)
        #self.assertEqual(vars(nopars), vars(rebuild))


    def test_set_attr(self):
       self.fail('to be implemented')

    def test_make_persistent(self):
        session.execute("DROP TABLE IF EXISTS hecuba_app.wordinfo")
        session.execute("DROP TABLE IF EXISTS hecuba_app.words")
        nopars = Words()
        self.assertFalse(nopars._persistent)
        config.batch_size = 1
        config.cache_activated = False
        nopars.ciao = 1
        nopars.ciao2 = "1"
        nopars.ciao3 = [1, 2, 3]
        nopars.ciao4 = (1, 2, 3)
        for i in range(10):
            nopars[i] = 'ciao'+str(i)

        count, = session.execute("SELECT count(*) FROM system_schema.tables WHERE table_name = 'hecuba_app' and keyspace_name = 'wordinfo'")[0]
        self.assertEqual(0, count)
        count, = session.execute("SELECT count(*) FROM system_schema.tables WHERE table_name = 'hecuba_app' and keyspace_name = 'words'")[0]
        self.assertEqual(0, count)

        nopars.make_persistent()

        count, = session.execute('SELECT count(*) FROM hecuba_app.wordinfo')[0]
        self.assertEqual(10, count)
        count, = session.execute('SELECT count(*) FROM hecuba_app.words')[0]
        self.assertEqual(4, count)




    def test_empty_persistent(self):
        session.execute("DROP TABLE IF EXISTS hecuba_app.wordinfo")
        session.execute("DROP TABLE IF EXISTS hecuba_app.words")
        from app.words import Words
        so = Words()
        so.make_persistent()
        so.ciao = "an attribute"
        so.another = 123
        config.batch_size = 1
        config.cache_activated = False
        for i in range(10):
            so[i] = str.join(',', map(lambda a: "ciao", range(i)))

        count, = session.execute('SELECT count(*) FROM hecuba_app.wordinfo')[0]
        self.assertEqual(10, count)
        count, = session.execute('SELECT count(*) FROM hecuba_app.words')[0]
        self.assertEqual(2, count)

        so.empty_persistent()

        count, = session.execute('SELECT count(*) FROM hecuba_app.wordinfo')[0]
        self.assertEqual(0, count)
        count, = session.execute('SELECT count(*) FROM hecuba_app.words')[0]
        self.assertEqual(0, count)


