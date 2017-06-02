import unittest

from hecuba import config
 

from hecuba.hdict import StorageDict


class PersistentDict_Tests(unittest.TestCase):

    @staticmethod
    def setUpClass():
        config.reset(mock_cassandra=False)

    def test_flush_items_cached(self):
        config.session.execute('DROP KEYSPACE IF EXISTS ksp')
        config.session.execute("CREATE KEYSPACE ksp WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")
        config.session.execute("CREATE TABLE ksp.tb1(pk1 int, val1 text,PRIMARY KEY(pk1))")
        config.cache_activated = True
        pd = StorageDict([('pk1', 'int')], [('val1', 'str')], 'ksp.tb1')
        config.batch_size = 101
        for i in range(100):
            pd[i] = 'ciao'+str(i)
        count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
        self.assertEqual(count, 0)
        pd._flush_items()
        count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
        self.assertEqual(count, 100)

    def test_persistent_nocache_batch10_setitem(self):
        config.session.execute('DROP KEYSPACE IF EXISTS ksp')
        config.session.execute("CREATE KEYSPACE ksp WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")
        config.session.execute("CREATE TABLE ksp.tb1(pk1 int, val1 text,PRIMARY KEY(pk1))")
        pd = StorageDict([('pk1', 'int')], [('val1', 'str')], 'ksp.tb1')
        config.batch_size = 10
        config.cache_activated = False
        for i in range(0, 9):
            pd[i] = 'fish'
        count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
        self.assertEqual(count, 0)
        pd[99] = 'fish'
        count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
        self.assertEqual(count, 10)

    def test_persistent_nocache_no_batch_setitem(self):
        config.session.execute('DROP KEYSPACE IF EXISTS ksp')
        config.session.execute(
            "CREATE KEYSPACE ksp WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")
        config.session.execute("CREATE TABLE ksp.tb1(pk1 int, val1 text,PRIMARY KEY(pk1))")
        pd = StorageDict([('pk1', 'int')], [('val1', 'str')], 'ksp.tb1')
        config.batch_size = 1
        config.cache_activated = False
        for i in range(0, 20):
            pd[i] = 'fish'
            count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
            self.assertEqual(count, i+1)

    def test_persistent_cache_batch10_setitem(self):
        config.session.execute('DROP KEYSPACE IF EXISTS ksp')
        config.session.execute(
            "CREATE KEYSPACE ksp WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")
        config.session.execute("CREATE TABLE ksp.tb1(pk1 int, val1 text,PRIMARY KEY(pk1))")
        pd = StorageDict([('pk1', 'int')], [('val1', 'str')], 'ksp.tb1')
        config.batch_size = 10
        config.cache_activated = True
        for i in range(0, 9):
            pd[i] = 'fish'
        count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
        self.assertEqual(count, 0)
        pd[99] = 'fish'
        count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
        self.assertEqual(count, 10)
