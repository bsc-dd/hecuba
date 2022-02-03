import unittest
import uuid

from mock import Mock
from .app.words import Words
from hecuba.IStorage import build_remotely


class MockStorageObj:
    pass


class BlockTest(unittest.TestCase):
    def test_astatic_creation(self):
        # TODO This test passes StorageDict arguments (results) to a StorageObj. Fix this test.

        results = {"built_remotely": False, "storage_id": uuid.uuid4(), "class_name": 'tests.app.words.Words',
                   "name": 'BlockTest.test_astatic_creation',
                   "columns": [('val1', 'str')], "entry_point": 'localhost', "primary_keys": [('pk1', 'int')],
                   "istorage_props": {}, "tokens": [(1, 2), (2, 3), (3, 4), (3, 5)]}

        words_mock_methods = Words._create_tables, Words._persist_attributes, Words._store_meta

        Words._create_tables = Mock(return_value=None)
        Words._persist_attributes = Mock(return_value=None)
        Words._store_meta = Mock(return_value=None)

        from hecuba import StorageDict
        sdict_mock_methods = StorageDict.make_persistent
        StorageDict.make_persistent = Mock(return_value=None)

        b = build_remotely(results)
        self.assertIsInstance(b, Words)
        Words._create_tables.assert_called_once()
        Words._persist_attributes.assert_called_once()
        assert (b._ksp == "BlockTest".lower())
        assert (b._table == b.__class__.__name__.lower())

        Words._create_tables, Words._persist_attributes, Words._store_meta = words_mock_methods
        StorageDict.make_persistent = sdict_mock_methods

    def test_iter_and_get_sets(self):
        """
        The iterator should read the same elements I can get with a __getitem__
        :return:
        """
        from hecuba.hdict import StorageDict
        b = StorageDict(None, [('pk1', 'str')], [('val', 'int')])

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
        bl.storage_id = u
        self.assertEquals(u, bl.storage_id)
        StorageDict.__init__ = old


if __name__ == '__main__':
    unittest.main()
