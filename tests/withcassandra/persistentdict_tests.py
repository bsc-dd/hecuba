import unittest

from hecuba import config, reset
 

from hecuba.dict import PersistentDict


class PersistentDict_Tests(unittest.TestCase):

    def setUp(self):
        reset()
    def flush_items_cached_test(self):
            config.session.execute('DROP KEYSPACE IF EXISTS ksp')
            config.session.execute("CREATE KEYSPACE ksp WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")
            config.session.execute("CREATE TABLE ksp.tb1(pk1 int, val1 text,PRIMARY KEY(pk1))")
            config.cache_activated = True
            pd = PersistentDict('ksp', 'tb1', True, [('pk1', 'int')], [('val1', 'str')])
            config.batch_size = 101
            for i in range(100):
                pd[i] = 'ciao'+str(i)
            count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
            self.assertEqual(count, 0)
            pd._flush_items()
            count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
            self.assertEqual(count, 100)

    def persistent_nocache_batch100_setitem_test(self):
        config.session.execute('DROP KEYSPACE IF EXISTS ksp')
        config.session.execute("CREATE KEYSPACE ksp WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")
        config.session.execute("CREATE TABLE ksp.tb1(pk1 int, val1 text,PRIMARY KEY(pk1))")
        pd = PersistentDict('ksp', 'tb1', True, [('pk1', 'int')], [('val1', 'str')])
        config.batch_size = 100
        config.cache_activated = False
        for i in range(0, 99):
            pd[i] = 'fish'
        count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
        self.assertEqual(count, 0)
        pd[99] = 'fish'
        count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
        self.assertEqual(count, 100)

    def persistent_nocache_no_batch_setitem_test(self):
        config.session.execute('DROP KEYSPACE IF EXISTS ksp')
        config.session.execute(
            "CREATE KEYSPACE ksp WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")
        config.session.execute("CREATE TABLE ksp.tb1(pk1 int, val1 text,PRIMARY KEY(pk1))")
        pd = PersistentDict('ksp', 'tb1', True, [('pk1', 'int')], [('val1', 'str')])
        config.batch_size = 1
        config.cache_activated = False
        for i in range(0, 20):
            pd[i] = 'fish'
            count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
            self.assertEqual(count, i+1)

    def persistent_cache_batch100_setitem_test(self):
        config.session.execute('DROP KEYSPACE IF EXISTS ksp')
        config.session.execute(
            "CREATE KEYSPACE ksp WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")
        config.session.execute("CREATE TABLE ksp.tb1(pk1 int, val1 text,PRIMARY KEY(pk1))")
        pd = PersistentDict('ksp', 'tb1', True, [('pk1', 'int')], [('val1', 'str')])
        config.batch_size = 100
        config.cache_activated = True
        for i in range(0, 99):
            pd[i] = 'fish'
        count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
        self.assertEqual(count, 0)
        pd[99] = 'fish'
        count, = config.session.execute('SELECT count(*) FROM ksp.tb1')[0]
        self.assertEqual(count, 100)
