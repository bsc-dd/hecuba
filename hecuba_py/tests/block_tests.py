import unittest
import uuid

from hecuba import Config
Config.reset(True)  ## THIS MUST STAY ONE THE TOP
from mock import Mock


class MockStorageObj:
    pass


class BlockTest(unittest.TestCase):
    @staticmethod
    def setUpClass():
        Config.reset(mock_cassandra=True)

    def test_astatic_creation(self):
        # TODO This test passes StorageDict arguments (results) to a StorageObj. Fix this test.
        class res: pass

        from app.words import Words
        results = res()
        results.storage_id = uuid.uuid4()
        results.class_name = 'tests.app.words.Words'
        results.name = 'ksp1.tab1'
        results.columns = [('val1', 'str')]
        results.entry_point = 'localhost'
        results.primary_keys = [('pk1', 'int')]
        results.istorage_props = {}
        results.tokens = [(1l, 2l), (2l, 3l), (3l, 4l), (3l, 5l)]

        words_mock_methods = Words._create_tables, Words._load_attributes, Words._store_meta

        Words._create_tables = Mock(return_value=None)
        Words._load_attributes = Mock(return_value=None)
        Words._store_meta = Mock(return_value=None)

        from hecuba import StorageDict
        sdict_mock_methods = StorageDict.make_persistent
        StorageDict.make_persistent = Mock(return_value=None)

        b = Words.build_remotely(results)
        self.assertIsInstance(b, Words)
        Words._create_tables.assert_called_once()
        Words._load_attributes.assert_called_once()
        Words._store_meta.assert_called_once()
        assert(b._ksp == "ksp1")
        assert (b._table == "tab1")

        Words._create_tables, Words._load_attributes, Words._store_meta = words_mock_methods
        StorageDict.make_persistent = sdict_mock_methods

    def test_iter_and_get_sets(self):
        """
        The iterator should read the same elements I can get with a __getitem__
        :return:
        """
        from hecuba.hdict import StorageDict
        b = StorageDict(None, [('pk1', 'str')], [('val', 'int')])
        b.is_persistent = False

        b['test1'] = 123124
        self.assertEqual(123124, b['test1'])

    def test_getID(self):
        """
        Checks that the id is the same
        :return:
        """
        from hecuba.hdict import StorageDict
        old = StorageDict.__init__
        StorageDict.__init__ = Mock(return_value=None)
        bl = StorageDict()
        u = uuid.uuid4()
        bl._storage_id = u
        self.assertEquals(str(u), bl.getID())
        StorageDict.__init__ = old


if __name__ == '__main__':
    unittest.main()