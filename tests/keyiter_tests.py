import unittest
from collections import defaultdict

from mock import Mock

from hecuba import config, reset
from hecuba.iter import KeyIter, Block
from random import shuffle


class KeyIterTest(unittest.TestCase):

    def setUp(self):
        reset()

    def test_calulate_block_ranges(self):
        nodes_to_tokens = dict([(i, 'localhost%d' % (i / 32)) for i in range(128)])
        res = KeyIter._calulate_block_ranges(nodes_to_tokens, 64)
        for el in res:
            self.assertEqual(len(el), 2)
            self.assertEqual(len(set(dict(el).values())), 1)

        tks = list(range(128))
        shuffle(tks)
        nodes_to_tokens = dict([(tks[i], 'localhost%d' % (i / 32)) for i in range(128)])
        res = KeyIter._calulate_block_ranges(nodes_to_tokens, 64)
        for el in res:
            self.assertEqual(len(el), 2)
            self.assertEqual(len(set(dict(el).values())), 1)

    def test_calulate_block_ODD_ranges(self):
        tks = list(range(128))
        shuffle(tks)
        nodes_to_tokens = dict([(tks[i], 'localhost%d' % (i / 10)) for i in range(128)])
        res = KeyIter._calulate_block_ranges(nodes_to_tokens, 64)
        self.assertEqual(len(res), 64)
        for el in res:
            self.assertEqual(len(el), 2)
            self.assertEqual(len(set(dict(el).values())), 1, "the block must refer a single node")
        count = defaultdict(int)
        for t, h in res:
            count[h[1]] += 1

        self.assertTrue(len(set(count.values())) <= 2)

    def test_next(self):
        tks = list(range(128))
        shuffle(tks)
        class fake:pass
        f = fake()
        config.number_of_blocks = 64
        f.token_to_host_owner = dict([(tks[i], 'localhost%d' % (i / 10)) for i in range(128)])

        f.metadata = f
        f.token_map = f
        config.cluster.metadata = f

        config.session.execute = Mock(return_value=None)

        ki = KeyIter('ksp1', 'tt1', 'hecuba.storageobj.StorageObj', ['pk1'])

        b = ki.next()
        self.assertIsInstance(b, Block)
        config.session.execute.assert_called()
