import unittest
import uuid

from mock import Mock

from hecuba import Config, config
from hecuba.hdict import StorageDict
from app.words import Words


class MockStorageObj:
    pass


class BlockTest(unittest.TestCase):
    @staticmethod
    def setUpClass():
        Config.reset(mock_cassandra=True)

    def test_static_creation(self):
        class res: pass

        results = res()
        results.blockid = u"aaaablockid"
        results.entry_point = u'localhost'
        results.dict_name = [u'pk1']
        results.tab = u"tab1"
        results.ksp = u'ksp1'
        results.tkns = [1l, 2l, 3l, 3l]
        results.storageobj_classname = u'app.words.Words'
        results.object_id = u'test_id'
        old = Words.__init__
        Words.__init__ = Mock(return_value=None)
        b = StorageDict.build_remotely(results)
        self.assertIsInstance(b.storageobj, Words)
        Words.__init__.assert_called_once_with("ksp1.tab1", storage_id='test_id')
        Words.__init__ = old

    def test_iter_and_get_sets(self):
        """
        The iterator should read the same elements I can get with a __getitem__
        :return:
        """
        b = StorageDict([('pk1', 'str')], [('val', 'int')])
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
