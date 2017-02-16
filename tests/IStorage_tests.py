import unittest

from hecuba import Config
from hecuba.IStorage import IStorage


class IStorageTest(unittest.TestCase):
    t16 = [(-9223372036854775808, -8070450532247928832),
           (-8070450532247928832, -6917529027641081856),
           (-6917529027641081856, -5764607523034234880),
           (-5764607523034234880, -4611686018427387904),
           (-4611686018427387904, -3458764513820540928),
           (-3458764513820540928, -2305843009213693952),
           (-2305843009213693952, -1152921504606846976),
           (-1152921504606846976, 0),
           (0, 1152921504606846976),
           (1152921504606846976, 2305843009213693952),
           (2305843009213693952, 3458764513820540928),
           (3458764513820540928, 4611686018427387904),
           (4611686018427387904, 5764607523034234880),
           (5764607523034234880, 6917529027641081856),
           (6917529027641081856, 8070450532247928832),
           (8070450532247928832, (2 ** 63) - 1)]

    @staticmethod
    def setUpClass():
        Config.reset(mock_cassandra=True)

    def tokens512_partition_16_nodes_test(self):
        partitions = [i for i in IStorage._tokens_partitions(self.t16, 512, 16)]
        flat = reduce(list.__add__, partitions)

        self.assertEqual(16, len(partitions))
        self.assertEqual((2 ** 63) - 1, reduce(max, map(lambda a: a[1], flat)))
        self.assertEqual(-9223372036854775808, reduce(min, map(lambda a: a[0], flat)))
        self.assertGreater(len(set(flat)), 512)

    def tokens16_partition_16_nodes_test(self):
        partitions = [i for i in IStorage._tokens_partitions(self.t16, 16, 16)]
        flat = reduce(list.__add__, partitions)

        self.assertEqual(16, len(partitions))
        self.assertEqual((2 ** 63) - 1, reduce(max, map(lambda a: a[1], flat)))
        self.assertEqual(-9223372036854775808, reduce(min, map(lambda a: a[0], flat)))
        self.assertEqual(16, len(set(flat)))
