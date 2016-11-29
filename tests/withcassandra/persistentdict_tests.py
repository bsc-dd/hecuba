import unittest

import hecuba
from hecuba.settings import session, config
from mock import Mock
from tests.block_tests import MockStorageObj

from hecuba.dict import PersistentDict


class PersistentDict_Tests(unittest.TestCase):
    def flush_items_cached_test(self):
            session.execute('DROP KEYSPACE IF EXISTS ksp')
            session.execute("CREATE KEYSPACE ksp WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")
            session.execute("CREATE TABLE ksp.tb1(pk1 int, val1 text,PRIMARY KEY(pk1))")
            config.cache_activated = True
            pd = PersistentDict('ksp', 'tb1', True, [('pk1', 'int')], [('val1', 'str')])
            config.batch_size = 101
            for i in range(100):
                pd[i] = 'ciao'+str(i)
            count, = session.execute('SELECT count(*) FROM ksp.tb1')[0]
            self.assertEqual(count, 0)
            pd._flush_items()
            count, = session.execute('SELECT count(*) FROM ksp.tb1')[0]
            self.assertEqual(count, 100)

    def persistent_nocache_batch100_setitem_test(self):
        pd = PersistentDict('ksp', 'tt', True, [('pk1', 'int')], [('val1', 'str')])
        config.batch_size = 100
        config.cache_activated = False
        session.execute = Mock(return_value=None)
        pd._flush_items = Mock(return_value=None)
        for i in range(0, 99):
            pd[i] = 'fish'
        pd._flush_items.assert_not_called()
        pd[99] = 'fish'
        pd._flush_items.assert_called_once()

    def persistent_cache_batch100_setitem_test(self):
        pd = PersistentDict('kksp', 'tt', True, [('pk1', 'int')], [('val1', 'str')])
        config.batch_size = 100
        config.cache_activated = True
        session.execute = Mock(return_value=None)
        pd._flush_items = Mock(return_value=None)
        for i in range(0, 99):
            pd[i] = 'fish'
        pd._flush_items.assert_not_called()
        pd[99] = 'fish'
        pd._flush_items.assert_not_called()

    def persistent_nocache_batch1_setitem_test(self):
        pd = PersistentDict('kksp', 'tt', True, [('pk1', 'int')], [('val1', 'str')])
        config.batch_size = 1
        config.cache_activated = False
        session.execute = Mock(return_value=None)
        pd._flush_items = Mock(return_value=None)
        pd[123] = 'fish'
        pd._flush_items.assert_called_once()