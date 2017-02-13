import unittest

from hecuba import config, Config
from mock import Mock


from hecuba.hdict import PersistentDict


class PersistentDict_Tests(unittest.TestCase):

    def setUp(self):
        Config.reset(mock_cassandra=True)


    #TODO to be written
    def test_end_prefetch(self):
        """
        self.prefetchManager.terminate()
        """



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
        pass

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
        ret = 'prepared-example'
        pd = PersistentDict('kksp', 'tt', True, [('pk1', 'int')], [('val1', 'str')])
        config.session.prepare = Mock(return_value=ret)
        self.assertEqual(ret, pd._preparequery('SELECT pk1,pk2 FROM kksp.tt WHERE pk1 = ?'))
        config.session.prepare.assert_called_once()
        self.assertEqual(ret, pd._preparequery('SELECT pk1,pk2 FROM kksp.tt WHERE pk1 = ?'))
        config.session.prepare.assert_called_once()
