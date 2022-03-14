import unittest
import gc

from hecuba import config
from hecuba.storageobj import StorageObj


class TestSimple(StorageObj):
    '''
    @ClassField words dict<<position:int>,value:str>
    '''
    pass


from .. import test_config
N_CASS_NODES = test_config.n_nodes


class StorageObjSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = config.execution_name
        config.execution_name = "StorageObjSplitTest".lower()

    @classmethod
    def tearDownClass(cls):
        #config.session.execute("DROP KEYSPACE IF EXISTS {}".format(config.execution_name))
        config.execution_name = cls.old

    # Create a new keyspace per test
    def setUp(self):
        self.ksp = config.execution_name

    def tearDown(self):
        pass

    def test_simple_keys_split_test(self):
        tablename = "tabskst"
        sto = TestSimple(tablename)
        pd = sto.words
        tbl_name = pd._table
        num_inserts = 1000
        what_should_be = set()
        for i in range(num_inserts):
            pd[i] = 'ciao' + str(i)
            what_should_be.add(i)
        del pd, sto

        gc.collect()
        count, = config.session.execute('SELECT count(*) FROM '+self.ksp+'.{}'.format(tbl_name))[0]
        self.assertEqual(count, num_inserts)

        sto = TestSimple(tablename)
        pd = sto.words

        count = 0
        res = set()
        splits = 0
        for partition in pd.split():
            splits += 1
            for val in partition.keys():
                res.add(val)
                count += 1
        pd.delete_persistent()
        del pd
        self.assertTrue(splits >= config.splits_per_node * N_CASS_NODES)
        self.assertEqual(count, num_inserts)
        self.assertEqual(what_should_be, res)

    def test_build_remotely_keys_split_test(self):
        tablename = 'tab30'
        sto = TestSimple(tablename)
        pd = sto.words
        tbl_name = pd._table
        num_inserts = 1000

        what_should_be = set()
        for i in range(num_inserts):
            pd[i] = 'ciao' + str(i)
            what_should_be.add(i)
        del pd, sto

        gc.collect()
        count, = config.session.execute('SELECT count(*) FROM '+self.ksp+'.{}'.format(tbl_name))[0]
        self.assertEqual(count, num_inserts)

        sto = TestSimple(tablename)
        pd = sto.words

        count = 0
        res = set()
        splits = 0
        for partition in pd.split():
            id = partition.storage_id
            from storage.api import getByID
            rebuild = getByID(id)
            splits += 1
            for val in rebuild.keys():
                res.add(val)
                count += 1
        pd.delete_persistent()
        del pd
        self.assertTrue(splits >= config.splits_per_node * N_CASS_NODES)
        self.assertEqual(count, num_inserts)
        self.assertEqual(what_should_be, res)

    #DISABLED TEST: Until split for storageobj has sense
    #def test_simple_keys_split_fromSO_test(self):
    #    tablename = "tab31"
    #    sto = TestSimple(tablename)
    #    pd = sto.words
    #    tbl_name = pd._table
    #    num_inserts = 1000
    #    what_should_be = set()
    #    for i in range(num_inserts):
    #        pd[i] = 'ciao' + str(i)
    #        what_should_be.add(i)
    #    del pd, sto

    #    gc.collect()
    #    count, = config.session.execute('SELECT count(*) FROM '+self.ksp+'.{}'.format(tbl_name))[0]
    #    self.assertEqual(count, num_inserts)

    #    sto = TestSimple(tablename)
    #    count = 0
    #    res = set()
    #    splits = 0
    #    for partition in sto.split():
    #        splits += 1
    #        for val in partition.words.keys():
    #            res.add(val)
    #            count += 1
    #    sto.delete_persistent()
    #    del sto
    #    self.assertTrue(splits >= config.splits_per_node * N_CASS_NODES)
    #    self.assertEqual(count, num_inserts)
    #    self.assertEqual(what_should_be, res)

    #DISABLED TEST: Until split for storageobj has sense
    #def test_build_remotely_keys_split_fromSO_test(self):
    #    tablename = "tab32"
    #    sto = TestSimple(tablename)
    #    pd = sto.words
    #    tbl_name = pd._table
    #    num_inserts = 1000
    #    what_should_be = set()
    #    for i in range(num_inserts):
    #        pd[i] = 'ciao' + str(i)
    #        what_should_be.add(i)
    #    del pd, sto

    #    gc.collect()
    #    count, = config.session.execute('SELECT count(*) FROM '+self.ksp+'.{}'.format(tbl_name))[0]
    #    self.assertEqual(count, num_inserts)

    #    sto = TestSimple(tablename)
    #    count = 0
    #    res = set()
    #    splits = 0
    #    for partition in sto.split():
    #        splits += 1
    #        id = partition.storage_id
    #        from storage.api import getByID
    #        rebuild = getByID(id)
    #        for val in rebuild.words.keys():
    #            res.add(val)
    #            count += 1
    #    sto.delete_persistent()
    #    del sto
    #    self.assertTrue(splits >= config.splits_per_node * N_CASS_NODES)
    #    self.assertEqual(count, num_inserts)
    #    self.assertEqual(what_should_be, res)

    def test_split_with_different_storage_ids(self):
        tablename = "tab33"
        sto = TestSimple(tablename)
        pd = sto.words

        ids = len(set(map(lambda x: x.storage_id, pd.split())))
        self.assertTrue(ids >= config.splits_per_node * N_CASS_NODES)
        sto.delete_persistent()


if __name__ == '__main__':
    unittest.main()
