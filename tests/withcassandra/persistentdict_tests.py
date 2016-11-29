import unittest

from mock import Mock
from tests.block_tests import MockStorageObj

from hecuba.dict import PersistentDict


class PersistentDict_Tests(unittest.TestCase):
    def flush_items_cached_test(self):
            pd = PersistentDict('ksp', 'tb1', True, [('pk1', 'int')], [('val1', 'str')])
            from hecuba.settings import config, session
            config.batch_size = 10
            session.execute = Mock(return_value=None)
            class Pre:pass
            ps = Pre()
            statement = Pre()
            ps.bind = Mock(return_value=statement)
            statement.query_string = Mock(return_value="SELECT TEST")
            session.prepare = Mock(return_value=ps)
            config.cache_activated = True
            for i in range(100):
                pd[i] = 'ciao'+str(i)
            session.execute.assert_not_called()
            pd._flush_items()
            session.execute.assert_any_call(10)

    def persistent_nocache_batch100_setitem_test(self):
        pd = PersistentDict('kksp', 'tt', True, [('pk1', 'int')], [('val1', 'str')])
        from hecuba.settings import config, session
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
        from hecuba.settings import config,session
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
        from hecuba.settings import config, session
        config.batch_size = 1
        config.cache_activated = False
        session.execute = Mock(return_value=None)
        pd._flush_items = Mock(return_value=None)
        pd[123] = 'fish'
        pd._flush_items.assert_called_once()