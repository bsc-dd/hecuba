import unittest

from hecuba import config
from hecuba.hdict import StorageDict


class StorageDictTest(unittest.TestCase):
    @staticmethod
    def setUpClass():
        config.session.execute(
            "CREATE KEYSPACE IF NOT EXISTS ksp WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")

    @staticmethod
    def tearDownClass():
        #config.session.execute( "DROP KEYSPACE ksp;")
        pass

    def tearDown(self):
        config.session.execute("DROP TABLE IF EXISTS ksp.tb1")

    def test_flush_items_100(self):
        config.session.execute("CREATE TABLE ksp.tb1(pk1 int, val1 text,PRIMARY KEY(pk1))")
        pd = StorageDict('ksp.tb1', [('pk1', 'int')], [('val1', 'text')])
        num_inserts = 100
        for i in range(num_inserts):
            pd[i] = 'ciao' + str(i)
        del pd  # To force hfetch to flush data
        import gc
        gc.collect()
        count, = config.session.execute("SELECT count(*) FROM ksp.tb1")[0]

        self.assertEqual(count, num_inserts)
        pd = StorageDict('ksp.tb1', [('pk1', 'int')], [('val1', 'text')])
        pd.delete_persistent()

    def test_flush_items_10K(self):
        config.session.execute("CREATE TABLE ksp.tb1(pk1 int, val1 text,PRIMARY KEY(pk1))")
        pd = StorageDict('ksp.tb1', [('pk1', 'int')], [('val1', 'text')])
        num_inserts = 10000
        for i in range(num_inserts):
            pd[i] = 'ciao' + str(i)
        del pd  # To force hfetch to flush data
        import gc
        gc.collect()
        count, = config.session.execute("SELECT count(*) FROM ksp.tb1 LIMIT " + str(num_inserts + 1))[0]
        self.assertEqual(count, num_inserts)
        pd = StorageDict('ksp.tb1', [('pk1', 'int')], [('val1', 'text')])
        pd.delete_persistent()

    '''
    def test_flush_items_1M(self):
        config.session.execute("DROP KEYSPACE IF EXISTS ksp")
        config.session.execute("CREATE KEYSPACE ksp WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")
        config.session.execute("CREATE TABLE ksp.tb1(pk1 int, val1 text,PRIMARY KEY(pk1))")
        pd = StorageDict('ksp.tb1', [('pk1', 'int')], [('val1', 'text')])
        num_inserts = 1000000
        for i in range(num_inserts):
            pd[i] = 'ciao' + str(i)
        del pd  # To force hfetch to flush data
        count, = config.session.execute("SELECT count(*) FROM ksp.tb1")[0]

        self.assertEqual(count, num_inserts)
    '''

    def test_write_and_then_read(self):
        config.session.execute("CREATE TABLE ksp.tb1(pk1 int, val1 text,PRIMARY KEY(pk1))")
        pd = StorageDict('ksp.tb1', [('pk1', 'int')], [('val1', 'text')])
        for i in range(100):
            pd[i] = 'ciao' + str(i)
        del pd  # To force hfetch to flush data
        import gc
        gc.collect()
        count, = config.session.execute("SELECT count(*) FROM ksp.tb1")[0]

        self.assertEqual(count, 100)
        pd = StorageDict('ksp.tb1', [('pk1', 'int')], [('val1', 'text')])
        for i in range(100):
            self.assertEqual(pd[i], u'ciao' + str(i))
        pd.delete_persistent()

    def test_write_and_then_read_named_tuple(self):
        config.session.execute("CREATE TABLE ksp.tb1(pk1 int, name text,age int,PRIMARY KEY(pk1))")
        pd = StorageDict('ksp.tb1', [('pk1', 'int')], [('name', 'text'), ('age', 'int')])
        for i in range(100):
            pd[i] = ['ciao' + str(i), i]
        del pd  # To force hfetch to flush data
        import gc
        gc.collect()
        count, = config.session.execute("SELECT count(*) FROM ksp.tb1")[0]

        self.assertEqual(count, 100)
        pd = StorageDict('ksp.tb1', [('pk1', 'int')], [('name', 'text'), ('age', 'int')])
        for i in range(100):
            name, age = pd[i]
            self.assertEqual(name, u'ciao' + str(i))
            self.assertEqual(age, i)

            self.assertEqual(pd[i].name, u'ciao' + str(i))
            self.assertEqual(pd[i].age, i)
        pd.delete_persistent()
