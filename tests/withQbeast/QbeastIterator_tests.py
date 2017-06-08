import unittest

from hecuba import config
from hecuba.hdict import StorageDict

from hecuba.qbeast import QbeastIterator, QbeastMeta


class IterThriftTests(unittest.TestCase):
    def test_read_without_split(self):
        it = QbeastIterator([('partid', 'int'), ('time', 'float')],
                            [('x', 'float'), ('y', 'float'), ('z', 'float')],
                            "test.particle", QbeastMeta('', [-.5, -.5, -.5], [3, 3, 3], 0.9))
        read = 0
        for _ in it:
            read += 1

        self.assertGreater(read, 0)
        print 'read ', read, 'elements'

    def test_read_with_split(self):

        it = QbeastIterator([('partid', 'int'), ('time', 'float')],
                            [('x', 'float'), ('y', 'float'), ('z', 'float')],
                            "test.particle", QbeastMeta('', [-.5, -.5, -.5], [3, 3, 3], .5))
        read = 0
        nparts = 0
        for partition in it.split():
            nparts += 1
            for _ in partition:
                read += 1
        self.assertGreater(read, 0)
        self.assertGreater(nparts, config.number_of_blocks)
        print 'read ', read, 'elements in splits :', nparts

    def test_read_with_split_remote(self):
        it = QbeastIterator([('partid', 'int'), ('time', 'float')],
                            [('x', 'float'), ('y', 'float'), ('z', 'float')],
                            "test.particle", QbeastMeta('', [-.5, -.5, -.5], [3, 3, 3], 0.9))
        read = 0
        nparts = 0
        for partition in it.split():
            nparts += 1
            rebuild = partition.build_remotely(partition._build_args)
            for _ in rebuild:
                read += 1
        self.assertGreater(read, 0)
        self.assertGreater(nparts, config.number_of_blocks)
        print 'read ', read, 'elements in splits :', nparts

    def test_same_data(self):
        it = QbeastIterator([('partid', 'int'), ('time', 'float')],
                            [('x', 'float'), ('y', 'float'), ('z', 'float')],
                            "test.particle", QbeastMeta('', [-.5, -.5, -.5], [3, 3, 3], 1))
        so = StorageDict([('partid', 'int'), ('time', 'float')],
                         [('x', 'float'), ('y', 'float'), ('z', 'float')],
                         "test.particle")

        read = 0
        for _ in it:
            read += 1

        read2 = 0
        for row in so.itervalues():
            if -.5 < row.x < 3 and -.5 < row.y < 3 and -.5 < row.z < 3:
                read2 += 1

        print read, ' vs ', read2
        self.assertEqual(read, read2)

