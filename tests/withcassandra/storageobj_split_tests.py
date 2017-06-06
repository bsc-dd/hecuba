import unittest

from hecuba import config
from hecuba.IStorage import IStorage
from hecuba.storageobj import StorageObj


class TestSimple(StorageObj):
    '''
    @ClassField words dict<<position:int>,value:str>
    '''
    pass

class StorageObjSplitTest(unittest.TestCase):

    def test_simple_iterkeys_split_test(self):
        # in process
        tablename = "tab30"
        sto = TestSimple("tab30")
        pd = sto.words

        what_should_be = set()
        for i in range(10000):
            pd[i] = 'ciao' + str(i)
            what_should_be.add(i)
        del pd, sto
        count, = config.session.execute('SELECT count(*) FROM hecuba.tab30_words')[0]
        self.assertEqual(count, 10000)

        sto = TestSimple("table30")
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
        self.assertTrue(splits >= config.number_of_blocks)
        self.assertEqual(count, 10000)
        self.assertEqual(what_should_be, res)

    def test_build_remotely_iterkeys_split_test(self):
        # in process
        tablename = "tab30"
        sto = TestSimple("tab30")
        pd = sto.words

        what_should_be = set()
        for i in range(10000):
            pd[i] = 'ciao' + str(i)
            what_should_be.add(i)
        del pd, sto
        count, = config.session.execute('SELECT count(*) FROM hecuba.tab30_words')[0]
        self.assertEqual(count, 10000)

        sto = TestSimple("table30")
        pd = sto.words

        count = 0
        res = set()
        splits = 0
        for partition in pd.split():
            id = partition.getID()
            rebuild = IStorage.getByID(id)
            splits += 1
            for val in rebuild.iterkeys():
                res.add(val)
                count += 1
        del pd
        self.assertTrue(splits >= config.number_of_blocks)
        self.assertEqual(count, 10000)
        self.assertEqual(what_should_be, res)

    def test_simple_iterkeys_split_fromSO_test(self):
        # in process
        sto = TestSimple("tab31")
        pd = sto.words

        what_should_be = set()
        for i in range(10000):
            pd[i] = 'ciao' + str(i)
            what_should_be.add(i)
        del pd, sto
        count, = config.session.execute('SELECT count(*) FROM hecuba.tab31_words')[0]
        self.assertEqual(count, 10000)

        sto = TestSimple("tab31")
        count = 0
        res = set()
        splits = 0
        for partition in sto.split():
            splits += 1
            for val in partition.words.iterkeys():
                res.add(val)
                count += 1
        del sto
        self.assertTrue(splits >= config.number_of_blocks)
        self.assertEqual(count, 10000)
        self.assertEqual(what_should_be, res)

    def test_build_remotely_iterkeys_split_fromSO_test(self):
        # in process
        sto = TestSimple("tab32")
        pd = sto.words

        what_should_be = set()
        for i in range(10000):
            pd[i] = 'ciao' + str(i)
            what_should_be.add(i)
        del pd, sto
        count, = config.session.execute('SELECT count(*) FROM hecuba.tab32_words')[0]
        self.assertEqual(count, 10000)

        sto = TestSimple("tab32")
        count = 0
        res = set()
        splits = 0
        for partition in sto.split():
            splits += 1
            id = partition.getID()
            rebuild = IStorage.getByID(id)
            for val in rebuild.words.iterkeys():
                res.add(val)
                count += 1
        del sto
        self.assertTrue(splits >= config.number_of_blocks)
        self.assertEqual(count, 10000)
        self.assertEqual(what_should_be, res)




if __name__ == '__main__':
    unittest.main()
