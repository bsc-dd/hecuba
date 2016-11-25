import unittest

from mock import Mock
from block_tests import MockStorageObj

from hecuba.dict import PersistentDict


class PersistentDict_Tests(unittest.TestCase):
    def init_test(self):
        self.fail('to be implemented')

    def init_prefetch_test(self):
        self.fail('to be implemented')

    def end_prefetch_test(self):
        self.fail('to be implemented')

    def contains__test(self):
        self.fail('to be implemented')

    def preparequery(self):
        self.fail('to be implemented')

    def iadd_test(self):
        self.fail('to be implemented')

    def setitem_test(self):
        self.fail('to be implemented')

    def getitem_test(self):
        self.fail('to be implemented')

    def writeitem_test(self):
        self.fail('to be implemented')

    def readitem_test(self):
        self.fail('to be implemented')

    def keys_test(self):
        self.fail('to be implemented')

    def buildquery_test(self):
        mypo = MockStorageObj()
        mypo.persistent = True
        mypo._table = "tt"
        mypo._ksp = "kksp"
        pd = PersistentDict(mypo, ['pk1', 'pk2'], ['val1', 'val2'])
        self.assertEqual('INSERT INTO kksp.tt(pk1,pk2,val1,val2) VALUES (?,?,?,?)', pd._build_insert_query())

    def buildcounterquery_test(self):
        mypo = MockStorageObj()
        mypo.persistent = True
        mypo._table = "tt"
        mypo._ksp = "kksp"
        pd = PersistentDict(mypo, ['pk1', 'pk2'], ['val1'])
        self.assertEqual('UPDATE kksp.tt SET val1 = val1 + ? WHERE pk1 = ? AND pk2 = ?', pd._build_insert_counter_query())

    def build_select_query_test(self):
        mypo = MockStorageObj()
        mypo.persistent = True
        mypo._table = "tt"
        mypo._ksp = "kksp"
        pd = PersistentDict(mypo, ['pk1', 'pk2'], ['val1'])
        self.assertEqual('SELECT pk1,pk2,val1 FROM kksp.tt WHERE pk1 = ?', pd._build_select_query(['pk1']))

        self.assertEqual('SELECT pk1,pk2,val1 FROM kksp.tt WHERE pk1 = ? AND pk2 = ?', pd._build_select_query(['pk1', 'pk2']))


    def persistent_nocache_nobatch_setitem_test(self):
        MockStorageObj.__init__ = Mock(return_value=None)
        mypo = MockStorageObj()
        mypo.persistent = True
        pd = PersistentDict(mypo, ['pk1'], ['val1', 'val2'])
        global batch_size, cache_activated
        batch_size = 1
        cache_activated = False
        from hecuba.settings import session
        session.execute = Mock(return_value=None)

        pd[123] = 'fish'

        session.execute.assert_called_once_with("INSERT INTO MockStorageObj(pk1, )")

    def inmemory_getitem_setitem_test(self):
        MockStorageObj.__init__ = Mock(return_value=None)
        mypo = MockStorageObj()
        mypo.persistent = False
        pd = PersistentDict(mypo, ['pk1', 'pk2'], ['val1', 'val2'])
        import random
        types = [random.randint(0, 100), random.random(),
                 float(random.random()), 'string_rand_'+str(random.random())
                 ]
        typeskeys = types[:]
        typeskeys.append([i for i in range(random.randint(0, 100))])
        typeskeys.append(False)

        for key in types:
            for value in typeskeys:
                pd[key] = value
                self.assertEqual(pd[key], value)






