import unittest

from mock import Mock

from hecuba import Config, config
from hecuba.iter import Block
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
        results.object_id =u'test_id'
        old = Words.__init__
        Words.__init__ = Mock(return_value=None)
        b = Block.build_remotely(results)
        self.assertIsInstance(b.storageobj, Words)
        Words.__init__.assert_called_once_with("ksp1.tab1", myuuid='test_id')
        Words.__init__ = old

    def test_init_creation(self):
        blockid = "aaaablockid"
        peer = 'localhost'
        tablename = "tab1"
        keyspace = 'ksp1'
        tokens = [1l, 2l, 3l, 3l]
        old = Words.__init__
        Words.__init__ = Mock(return_value=None)
        b = Block(blockid, peer, tablename, keyspace, tokens, 'app.words.Words')
        self.assertIsInstance(b.storageobj, Words)
        Words.__init__.assert_called_once_with(keyspace + "." + tablename, myuuid=None)
        Words.__init__ = old

    def test_iter_and_get_sets(self):
        """
        The iterator should read the same elements I can get with a __getitem__
        :return:
        """
        blockid = "aaaablockid"
        peer = 'localhost'
        tablename = "tab1"
        keyspace = 'ksp1'
        tokens = [1l, 2l, 3l, 3l]
        b = Block(blockid, peer, tablename, keyspace, tokens, 'app.words.Words')
        b.storageobj._get_default_dict().is_persistent = False
        self.assertIsInstance(b.storageobj, Words)

        b['test1'] = 123124
        self.assertEqual(123124, b['test1'])


    def test_getID(self):
        """
        Check the id is the same
        :return:
        """
        from hecuba.iter import Block
        old = Block.__init__
        Block.__init__ = Mock(return_value=None)
        bl = Block()
        bl.blockid = 'myuuid'
        self.assertEquals('myuuid', bl.getID())
        self.assertNotEquals('myuuid2', bl.getID())
        Block.__init__ = old


if __name__ == '__main__':
    unittest.main()
