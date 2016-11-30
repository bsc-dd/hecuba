import unittest

import hecuba
from mock import Mock
from block_tests import MockStorageObj

from hecuba.dict import PersistentDict


class PersistentDict_Tests(unittest.TestCase):
    '''
    def test_init(self):
        self.fail('to be implemented')

    def test_init_prefetch(self):
        self.fail('to be implemented')

    def test_end_prefetch(self):
        self.fail('to be implemented')

    def test_ddbb_contains(self):
        self.fail('to be implemented')

    def test_iadd(self):
        self.fail('to be implemented')

    def test_getitem(self):
        self.fail('to be implemented')

    def test_ddbb_getitem(self):
        self.fail('to be implemented')

    def test_ddbb_writeitem(self):
        self.fail('to be implemented')

    def test_inmemory_writeitem(self):
        self.fail('to be implemented')

    def test_ddbb_readitem(self):
        MockStorageObj.__init__ = Mock(return_value=None)
        mypo = MockStorageObj()
        mypo._ksp = "kksp"
        mypo._table = "tt"
        pd = PersistentDict(mypo, ['pk1'], ['pk2'])
        pd.mypo.persistent = True
        pd._flush_items = Mock(return_value=None)
        from hecuba import session
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

    def test_readitem(self):
        self.fail('to be implemented')
    '''

    def test_init_prefetch(self):
        """
        self.prefetch = True
        self.prefetchManager = PrefetchManager(1, 1, block)
        """
        pd = PersistentDict('ksp', 'tb1', True, [('pk1', 'int')], [('val1', 'str')])
        pd.prefetch = False
        from hecuba.prefetchmanager import PrefetchManager
        PrefetchManager.__init__ = Mock(return_value=None)
        from hecuba.iter import Block
        Block.__init__ = Mock(return_value=None)
        pd.init_prefetch(Block())
        PrefetchManager.__init__.assert_called_once()
        self.assertEqual(True, pd.prefetch)

    def test_end_prefetch(self):
        """
        self.prefetchManager.terminate()
        """
        from hecuba import session
        session.execute = Mock(return_value=None)

        class mockme: pass

        mm = mockme()
        mm.bind = Mock(return_value=None)
        session.prepare = Mock(return_value=mm)
        pd = PersistentDict('ksp', 'tb1', False, [('pk1', 'int')], [('val1', 'str')])
        pd.prefetch = True

        from hecuba.iter import Block
        bl = Block('myuuid', 'localhost', ['pk1'], 'tt', 'ksp', [1, 2], 'app.words.Words')
        pd.init_prefetch(bl)

    def test_iadd(self):
        pd = PersistentDict('ksp', 'tb1', False, [('pk1', 'int')], [('val1', 'int')])
        pd.types['pk1'] = "notCounter"
        pd[3] = 0
        pd[3] += 2
        self.assertEqual(2, pd[3])
        self.assertNotEquals(1, pd[3])
        pd2 = PersistentDict('ksp', 'tb1', False, [('pk1', 'int')], [('pk2', 'counter')])
        pd2[3] = 3
        pd2[3] += 2
        self.assertEqual(2, pd[3])
        self.assertNotEquals(5, pd[3])

    def test_init(self):
        from hecuba.cache import PersistentDictCache
        PersistentDictCache.__init__ = Mock(return_value=None)
        pd = PersistentDict('ksp', 'tb1', False, [('pk1', 'int')], [('val1', 'str')])
        pd.dictCache.__init__.assert_called_once()

    def inmemory_contains_test(self):
        pd = PersistentDict('ksp', 'tb1', False, [('pk1', 'int')], [('val1', 'str')])
        pd[3] = '4'
        self.assertEqual(True, 3 in pd)
        self.assertEqual('4', pd[3])

    def inmemory_keys_test(self):
        pd = PersistentDict('ksp', 'tb1', False, [('pk1', 'int')], [('val1', 'str')])
        pd[0] = '1'
        pd[1] = '2'
        pd[2] = '3'
        pd[3] = '4'
        self.assertEqual([0, 1, 2, 3], pd.keys())

    def buildquery_test(self):
        pd = PersistentDict('kksp', 'tt', False, [('pk1', 'int'), ('pk2', 'int')], [('val1', 'str'), ('val2', 'str')])
        self.assertEqual('INSERT INTO kksp.tt(pk1,pk2,val1,val2) VALUES (?,?,?,?)', pd._build_insert_query())

    def buildcounterquery_test(self):
        pd = PersistentDict('kksp', 'tt', False, [('pk1', 'int'), ('pk2', 'int')], [('val1', 'str')])
        self.assertEqual('UPDATE kksp.tt SET val1 = val1 + ? WHERE pk1 = ? AND pk2 = ?',
                         pd._build_insert_counter_query())

        pd = PersistentDict('kksp', 'tt', False, [('pk1', 'int')], [('val1', 'str')])
        self.assertEqual('UPDATE kksp.tt SET val1 = val1 + ? WHERE pk1 = ?', pd._build_insert_counter_query())

    def build_select_query_test(self):
        pd = PersistentDict('kksp', 'tt', True, [('pk1', 'int'), ('pk2', 'int')], [('val1', 'str')])
        self.assertEqual('SELECT pk1,pk2,val1 FROM kksp.tt WHERE pk1 = ?', pd._build_select_query(['pk1']))
        self.assertEqual('SELECT pk1,pk2,val1 FROM kksp.tt WHERE pk1 = ? AND pk2 = ?',
                         pd._build_select_query(['pk1', 'pk2']))

    def inmemory_getitem_setitem_test(self):
        pd = PersistentDict('kksp', 'tt', False, [('pk1', 'int'), ('pk2', 'int')], [('val1', 'str'), ('val2', 'str')])
        import random
        types = [random.randint(0, 100), random.random(),
                 float(random.random()), 'string_rand_' + str(random.random())
                 ]
        typeskeys = types[:]
        typeskeys.append([i for i in range(random.randint(0, 100))])
        typeskeys.append(False)

        for key in types:
            for value in typeskeys:
                pd[key] = value
                self.assertEqual(pd[key], value)

    def preparequery_test(self):
        from hecuba import session
        ret = 'prepared-example'
        pd = PersistentDict('kksp', 'tt', True, [('pk1', 'int')], [('val1', 'str')])
        session.prepare = Mock(return_value=ret)
        self.assertEqual(ret, pd._preparequery('SELECT pk1,pk2 FROM kksp.tt WHERE pk1 = ?'))
        session.prepare.assert_called_once()
        self.assertEqual(ret, pd._preparequery('SELECT pk1,pk2 FROM kksp.tt WHERE pk1 = ?'))
        session.prepare.assert_called_once()
