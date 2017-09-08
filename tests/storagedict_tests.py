import unittest

from hecuba import Config
from hecuba.hdict import StorageDict


class StorageDict_Tests(unittest.TestCase):
    def setUp(self):
        Config.reset(mock_cassandra=True)

    def test_init(self):
        pass

    def inmemory_contains_test(self):
        pd = StorageDict(None,
                         [('pk1', 'int')],
                         [('val1', 'str')])
        pd[3] = '4'
        self.assertEqual(True, 3 in pd)
        self.assertEqual('4', pd[3])

    def inmemory_keys_test(self):
        pd = StorageDict(None,
                         [('pk1', 'int'), ('pk2', 'int')],
                         [('val1', 'str')])

        pd[0] = '1'
        pd[1] = '2'
        pd[2] = '3'
        pd[3] = '4'
        self.assertEqual({0, 1, 2, 3}, set(pd.keys()))

    def inmemory_composed_keys_test(self):
        pd = StorageDict(None,
                         [('pk1', 'int'), ('pk2', 'int')],
                         [('val1', 'str')])

        pd[0, 1] = '1'
        pd[1, 1] = '2'
        pd[2, 0] = '3'
        pd[3, 1] = '4'
        self.assertEqual({(0, 1), (1, 1), (2, 0), (3, 1)}, set(pd.keys()))

    def inmemory_getitem_setitem_test(self):
        pd = StorageDict(None,
                         [('pk1', 'int'), ('pk2', 'int')],
                         [('val1', 'str')])

        import random
        types = [random.randint(0, 100), random.random(),
                 float(random.random()), 'string_rand_' + str(random.random())
                 ]
        typeskeys = types[:]
        typeskeys.append([i for i in range(random.randint(0, 100))])
        typeskeys.append(False)

        for key in types:
            for value in typeskeys:
                pd[key] = value
                self.assertEqual(pd[key], value)
