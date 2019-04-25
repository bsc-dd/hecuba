import unittest

from hecuba import config
from hecuba.storageobj import StorageObj


class TestSimple(StorageObj):
    '''
    @ClassField words dict<<position:int>,value:str>
    '''
    pass


class StorageObjSplitTest(unittest.TestCase):
    def test_simple_iterkeys_split_test(self):
        tablename = "tab30"
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename + "_words")
        sto = TestSimple(tablename)
        pd = sto.words
        num_inserts = 1000
        what_should_be = set()
        for i in range(num_inserts):
            pd[i] = 'ciao' + str(i)
            what_should_be.add(i)
        del pd, sto
        count, = config.session.execute('SELECT count(*) FROM my_app.' + tablename + '_words')[0]
        self.assertEqual(count, num_inserts)

        sto = TestSimple(tablename)
        pd = sto.words

        count = 0
        res = set()
        splits = 0
        for partition in pd.split():
            splits += 1
            for val in partition.iterkeys():
                res.add(val)
                count += 1
        del pd
        self.assertTrue(splits >= config.number_of_partitions)
        self.assertEqual(count, num_inserts)
        self.assertEqual(what_should_be, res)

    def test_build_remotely_iterkeys_split_test(self):
        tablename = 'tab30'
        config.session.execute('DROP TABLE IF EXISTS my_app.' + tablename)
        config.session.execute('DROP TABLE IF EXISTS my_app.' + tablename + '_words')
        sto = TestSimple(tablename)
        pd = sto.words
        num_inserts = 1000

        what_should_be = set()
        for i in range(num_inserts):
            pd[i] = 'ciao' + str(i)
            what_should_be.add(i)
        del pd, sto
        count, = config.session.execute('SELECT count(*) FROM my_app.' + tablename + '_words')[0]
        self.assertEqual(count, num_inserts)

        sto = TestSimple(tablename)
        pd = sto.words

        count = 0
        res = set()
        splits = 0
        for partition in pd.split():
            id = partition.getID()
            from storage.api import getByID
            rebuild = getByID(id)
            splits += 1
            for val in rebuild.iterkeys():
                res.add(val)
                count += 1
        del pd
        self.assertTrue(splits >= config.number_of_partitions)
        self.assertEqual(count, num_inserts)
        self.assertEqual(what_should_be, res)

    def test_simple_iterkeys_split_fromSO_test(self):
        tablename = "tab31"
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename + "_words")
        sto = TestSimple(tablename)
        pd = sto.words
        num_inserts = 1000
        what_should_be = set()
        for i in range(num_inserts):
            pd[i] = 'ciao' + str(i)
            what_should_be.add(i)
        del pd, sto
        count, = config.session.execute('SELECT count(*) FROM my_app.' + tablename + '_words')[0]
        self.assertEqual(count, num_inserts)

        sto = TestSimple(tablename)
        count = 0
        res = set()
        splits = 0
        for partition in sto.split():
            splits += 1
            for val in partition.words.iterkeys():
                res.add(val)
                count += 1
        del sto
        self.assertTrue(splits >= config.number_of_partitions)
        self.assertEqual(count, num_inserts)
        self.assertEqual(what_should_be, res)

    def test_build_remotely_iterkeys_split_fromSO_test(self):
        tablename = "tab32"
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename + "_words")
        sto = TestSimple(tablename)
        pd = sto.words
        num_inserts = 1000
        what_should_be = set()
        for i in range(num_inserts):
            pd[i] = 'ciao' + str(i)
            what_should_be.add(i)
        del pd, sto
        count, = config.session.execute('SELECT count(*) FROM my_app.' + tablename + '_words')[0]
        self.assertEqual(count, num_inserts)

        sto = TestSimple(tablename)
        count = 0
        res = set()
        splits = 0
        for partition in sto.split():
            splits += 1
            id = partition.getID()
            from storage.api import getByID
            rebuild = getByID(id)
            for val in rebuild.words.iterkeys():
                res.add(val)
                count += 1
        del sto
        self.assertTrue(splits >= config.number_of_partitions)
        self.assertEqual(count, num_inserts)
        self.assertEqual(what_should_be, res)

    def test_split_with_different_storage_ids(self):
        tablename = "tab32"
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename)
        config.session.execute("DROP TABLE IF EXISTS my_app." + tablename + "_words")
        sto = TestSimple(tablename)
        pd = sto.words

        ids = len(set(map(lambda x: x._storage_id, pd.split())))
        self.assertTrue(ids >= config.number_of_partitions)


if __name__ == '__main__':
    unittest.main()
