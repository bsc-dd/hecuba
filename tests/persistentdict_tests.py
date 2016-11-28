import unittest

from mock import Mock
from block_tests import MockStorageObj

from hecuba.dict import PersistentDict


class PersistentDict_Tests(unittest.TestCase):
    '''

    def test_ddbb_contains(self):
        self.fail('to be implemented')

    def test_ddbb_getitem(self):
        self.fail('to be implemented')

    def test_ddbb_writeitem(self):
        self.fail('to be implemented')

    def test_ddbb_readitem(self):
        MockStorageObj.__init__ = Mock(return_value=None)
        mypo = MockStorageObj()
        mypo._ksp = "kksp"
        mypo._table = "tt"
        pd = PersistentDict(mypo, ['pk1'], ['pk2'])
        pd.mypo.persistent = True
        pd._flush_items = Mock(return_value=None)
        from hecuba.settings import session
        class MyPS:pass
        ps = MyPS()
        class MyStatement:pass
        st=MyStatement()
        ps.bind = Mock(return_value=st)
        session.prepare = Mock(return_value=ps)
        pd._readitem([3])
        session.prepare.assert_called_once()
        ps.bind.assert_called_with(3)
        session.execute.assert_called_with(st)

    '''

    def test_init_prefetch(self):
        """
        self.prefetch = True
        self.prefetchManager = PrefetchManager(1, 1, block)
        """
        MockStorageObj.__init__ = Mock(return_value=None)
        mypo = MockStorageObj()
        pd = PersistentDict(mypo, ['pk1'], ['val1'])
        pd.prefetch = False
        from hecuba.prefetchmanager import PrefetchManager
        PrefetchManager.__init__ = Mock(return_value=None)
        from hecuba.iter import Block
        Block.__init__ = Mock(return_value=None)
        pd.init_prefetch(Block())
        PrefetchManager.__init__.assert_called_once()
        self.assertEqual(True, pd.prefetch)
    '''
    def test_end_prefetch(self):
        """
        self.prefetchManager.terminate()
        """
        MockStorageObj.__init__ = Mock(return_value=None)
        mypo = MockStorageObj()
        pd = PersistentDict(mypo, ['pk1'], ['val1'])
        pd.KeyList = {'mydict':'myName'}
        pd.prefetch = True
        from hecuba.iter import Block
        Block.__init__ = Mock(return_value=None)
        bl = Block()
        bl.key_names = 'ksp'
        bl.peer = "localhost"
        bl.blockid = 'myuuid'
        bl.keyspace = 'ksp'
        bl.table_name = 'tt'
        bl.token_ranges = [1, 2]
        bl.storageobj = pd
        pd.init_prefetch(bl)
        pd.prefetchmanager.__init__ = Mock(return_value=None)
        pd.prefetchManager.terminate = Mock(return_value=None)
        pd.end_prefetch()
        pd.prefetchManager.terminate.assert_called_once()
    '''
    def test_iadd(self):
        MockStorageObj.__init__ = Mock(return_value=None)
        mypo = MockStorageObj()
        mypo.persistent = False
        pd = PersistentDict(mypo, ['pk1'], ['val1'])
        pd.types['pk1'] = "notCounter"
        pd[3] = 0
        pd[3] += 2
        self.assertEqual(2, pd[3])
        self.assertNotEquals(1, pd[3])
        pd2 = PersistentDict(mypo, ['pk2'], ['val2'])
        pd2.types['pk2'] = "counter"
        pd2[3] = 3
        pd2[3] += 2
        self.assertEqual(2, pd[3])
        self.assertNotEquals(5, pd[3])

    def test_init(self):
        MockStorageObj.__init__ = Mock(return_value=None)
        mypo = MockStorageObj()
        from hecuba.cache import PersistentDictCache
        PersistentDictCache.__init__ = Mock(return_value=None)
        pd = PersistentDict(mypo, ['pk1'], ['val1'])
        pd.dictCache.__init__.assert_called_once()

    def test_inmemory_contains(self):
        MockStorageObj.__init__ = Mock(return_value=None)
        mypo = MockStorageObj()
        pd = PersistentDict(mypo, ['pk1'], ['val1'])
        pd.mypo.persistent = False
        pd.mypo.__getitem__ = Mock(return_value=None)
        pd[3] = '4'
        #print "type(pd):", type(pd)
        self.assertEqual(True, pd.__contains__(3))
        pd.mypo.__getitem__.assert_not_called()
        self.assertEqual(False, pd.__contains__(33))
        pd.mypo.__getitem__.assert_not_called()

    def test_inmemory_keys(self):
        MockStorageObj.__init__ = Mock(return_value=None)
        mypo = MockStorageObj()
        mypo.persistent = False
        pd = PersistentDict(mypo, ['pk1'], ['val1'])
        pd[0] = '1'
        pd[1] = '2'
        pd[2] = '3'
        pd[3] = '4'
        self.assertEqual([0,1,2,3], pd.keys())

    def test_buildquery(self):
        mypo = MockStorageObj()
        mypo.persistent = True
        mypo._table = "tt"
        mypo._ksp = "kksp"
        pd = PersistentDict(mypo, ['pk1', 'pk2'], ['val1', 'val2'])
        self.assertEqual('INSERT INTO kksp.tt(pk1,pk2,val1,val2) VALUES (?,?,?,?)', pd._build_insert_query())

    def test_buildcounterquery(self):
        mypo = MockStorageObj()
        mypo.persistent = True
        mypo._table = "tt"
        mypo._ksp = "kksp"
        pd = PersistentDict(mypo, ['pk1', 'pk2'], ['val1'])
        self.assertEqual('UPDATE kksp.tt SET val1 = val1 + ? WHERE pk1 = ? AND pk2 = ?', pd._build_insert_counter_query())

    def test_build_select_query(self):
        mypo = MockStorageObj()
        mypo.persistent = True
        mypo._table = "tt"
        mypo._ksp = "kksp"
        pd = PersistentDict(mypo, ['pk1', 'pk2'], ['val1'])
        self.assertEqual('SELECT pk1,pk2,val1 FROM kksp.tt WHERE pk1 = ?', pd._build_select_query(['pk1']))
        self.assertEqual('SELECT pk1,pk2,val1 FROM kksp.tt WHERE pk1 = ? AND pk2 = ?', pd._build_select_query(['pk1', 'pk2']))

    def persistent_nocache_batch1_setitem_test(self):
        MockStorageObj.__init__ = Mock(return_value=None)
        mypo = MockStorageObj()
        mypo.persistent = True
        pd = PersistentDict(mypo, ['pk1'], ['val1', 'val2'])
        from hecuba.settings import config,session
        config.batch_size = 1
        cache_activated = False
        session.execute = Mock(return_value=None)
        pd._flush_items = Mock(return_value=None)
        pd[123] = 'fish'
        pd._flush_items.assert_called_once()

    def persistent_nocache_batch100_setitem_test(self):
        MockStorageObj.__init__ = Mock(return_value=None)
        mypo = MockStorageObj()
        mypo.persistent = True
        pd = PersistentDict(mypo, ['pk1'], ['val1', 'val2'])
        from hecuba.settings import config,session
        config.batch_size = 100
        cache_activated = False
        session.execute = Mock(return_value=None)
        pd._flush_items = Mock(return_value=None)
        for i in range(0,99):
            pd[i] = 'fish'
        pd._flush_items.assert_not_called()
        pd[99] = 'fish'
        pd._flush_items.assert_called_once()

    def persistent_cache_batch100_setitem_test(self):
        MockStorageObj.__init__ = Mock(return_value=None)
        mypo = MockStorageObj()
        mypo.persistent = True
        pd = PersistentDict(mypo, ['pk1'], ['val1', 'val2'])
        from hecuba.settings import config,session
        config.batch_size = 100
        cache_activated = True
        session.execute = Mock(return_value=None)
        pd._flush_items = Mock(return_value=None)
        for i in range(0,99):
            pd[i] = 'fish'
        pd._flush_items.assert_not_called()
        pd[99] = 'fish'
        pd._flush_items.assert_called_once()

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

    def preparequery_test(self):
        from hecuba.settings import session
        ret = 'prepared-example'
        session.prepare = Mock(return_value=ret)
        MockStorageObj.__init__ = Mock(return_value=None)
        mypo = MockStorageObj()
        pd = PersistentDict(mypo, ['pk1'], ['val1'])
        self.assertEqual(ret,pd._preparequery('SELECT pk1,pk2 FROM kksp.tt WHERE pk1 = ?'))
        session.prepare.assert_called_once()
        self.assertEqual(ret,pd._preparequery('SELECT pk1,pk2 FROM kksp.tt WHERE pk1 = ?'))
        session.prepare.assert_called_once()

if __name__ == '__main__':
    unittest.main()
