import unittest

from mock import Mock

from hecuba.iter import Block


class MockStorageObj:
   pass


class BlockTest(unittest.TestCase):
    def static_creation_test(self):
        class res: pass

        results = res()
        results.blockid = "aaaablockid"
        results.entry_point = 'localhsot'
        results.dict_name = ['pk1']
        results.tab = "tab1"
        results.ksp = 'ksp1'
        results.tkns = [1l, 2l, 3l, 3l]
        results.storageobj_classname = 'block_tests.MockStorageObj'

        MockStorageObj.__init__ = Mock(return_value=None)
        b = Block.build_remotely(results)
        self.assertIsInstance(b.storageobj, MockStorageObj)
        MockStorageObj.__init__.assert_called_once_with(ksp=results.ksp, table=results.tab)
        self.fail('to be implemented')

    def init_creation_test(self):
        blockid = "aaaablockid"
        peer = 'localhsot'
        keynames = ['pk1']
        tablename = "tab1"
        keyspace = 'ksp1'
        tokens = [1l, 2l, 3l, 3l]
        MockStorageObj.__init__ = Mock(return_value=None)
        b = Block(blockid, peer, keynames, tablename, keyspace, tokens, 'block_tests.MockStorageObj')
        self.assertIsInstance(b.storageobj, MockStorageObj)
        MockStorageObj.__init__.assert_called_once_with(ksp=keyspace, table=tablename)

    def itering_test(self):
        """
        Block should be iterable and return in BlockIter
        :return:
        """
        self.fail('to be implemented')

    def get_and_set_item_test(self):
        self.fail('to be implemented')

    def iter_and_get_sets_test(self):
        """
        The iterator should read the same elements I can get with a __getitem__
        :return:
        """
        blockid = "aaaablockid"
        peer = 'localhsot'
        keynames = ['pk1']
        tablename = "tab1"
        keyspace = 'ksp1'
        tokens = [1l, 2l, 3l, 3l]
        MockStorageObj.__init__ = Mock(return_value=None)
        b = Block(blockid, peer, keynames, tablename, keyspace, tokens, 'block_tests.MockStorageObj')
        self.assertIsInstance(b.storageobj, MockStorageObj)
        MockStorageObj.__init__.assert_called_once_with(ksp=keyspace, table=tablename)
        b['test1'] = 123124
        self.assertEqual(123124, b['test1'])

    def getID_test(self):
        """
        Check the id is the same
        :return:
        """
        self.fail('to be implemented')

    def iteritems_test(self):
        self.fail('to be implemented')

    def itervalues(self):
        self.fail('to be implemented')

    def iterkeys(self):
        self.fail('to be implemented')
