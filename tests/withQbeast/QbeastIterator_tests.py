import unittest

from hecuba import config
from hecuba.qbeast import QbeastIterator, QbeastMeta
from tests.withQbeast import TestServers


class IterThriftTests(unittest.TestCase):
    def setUp(self):
        self.ss = TestServers(100)

    def tearDown(self):
        self.ss.stopServers()

    def test_can_be_rebuild(self):
        it = QbeastIterator([('partid', 'int'), ('time', 'float')],
                            [('x', 'float'), ('y', 'float'), ('z', 'float')],
                            "test.particle", QbeastMeta('', [-.5, -.5, -.5], [3, 3, 3], 0.9))
        from storage.api import getByID
        it2 = getByID(it.getID())
        self.assertEqual(it.getID(), it2.getID())

    def test_read_without_split(self):
        it = QbeastIterator([('partid', 'int'), ('time', 'float')],
                            [('x', 'float'), ('y', 'float'), ('z', 'float')],
                            "test.particle", QbeastMeta('', [-.5, -.5, -.5], [3, 3, 3], 0.9),
                            entry_point='localhost'
                            )
        read = 0
        for _ in it:
            read += 1

        self.assertEqual(read, 100)

    def test_read_with_split(self):
        it = QbeastIterator([('partid', 'int'), ('time', 'float')],
                            [('x', 'float'), ('y', 'float'), ('z', 'float')],
                            "test.particle", QbeastMeta('', [-.5, -.5, -.5], [3, 3, 3], .5),
                            entry_point='localhost')
        read = 0
        nparts = 0
        for partition in it.split():
            nparts += 1
            for _ in partition:
                read += 1
        self.assertGreater(read, 100)
        self.assertGreater(nparts, config.number_of_blocks)

    def test_read_with_split_remote(self):
        it = QbeastIterator([('partid', 'int'), ('time', 'float')],
                            [('x', 'float'), ('y', 'float'), ('z', 'float')],
                            "test.particle", QbeastMeta('', [-.5, -.5, -.5], [3, 3, 3], 0.9),
                            entry_point='localhost')
        read = 0
        nparts = 0
        for partition in it.split():
            nparts += 1
            rebuild = partition.build_remotely(partition._build_args)
            for _ in rebuild:
                read += 1
        self.assertEqual(read, 100 * nparts)
        self.assertGreater(nparts, config.number_of_blocks)
