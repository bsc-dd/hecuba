import unittest
from collections import defaultdict

from hecuba import config, StorageDict
from hecuba.IStorage import _discrete_token_ranges
from hecuba.partitioner import Partitioner

from .. import test_config


class SimpleObj(StorageDict):
    '''
    @TypeSpec dict<<k:int>, v:int>
    '''


class IStorageSplitLocalityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.session.execute("CREATE KEYSPACE IF NOT EXISTS test_ksp  WITH replication = "
                               "{'class': 'SimpleStrategy', 'replication_factor': 1};")

        config.session.execute("CREATE TABLE IF NOT EXISTS test_ksp.tab(k int PRIMARY KEY,v int)")

    def test_enough_token(self):
        obj = SimpleObj("test_ksp.tab")
        partitioner = Partitioner(obj, "SIMPLE")
        config.splits_per_node = 10
        tkns_p = list(partitioner._tokens_partitions("test_ksp", "tab", token_range_size=None,
                                                     target_token_range_size=64 * 1024 * 1024))
        tkns_p = [i[1] for i in tkns_p]
        self.check_all(tkns_p, 10, 20)

    def test_too_little_tokens(self):
        obj = SimpleObj("test_ksp.tab")
        partitioner = Partitioner(obj, "SIMPLE")
        config.splits_per_node = 1000
        tkns_p = list(partitioner._tokens_partitions("test_ksp", "tab", token_range_size=None,
                                                     target_token_range_size=64 * 1024))
        tkns_p = [i[1] for i in tkns_p]
        self.check_all(tkns_p, 1000, 1000)

    def test_splitting_tokens(self):
        obj = SimpleObj("test_ksp.tab")
        partitioner = Partitioner(obj, "SIMPLE")
        config.splits_per_node = 1
        tkns_p = list(partitioner._tokens_partitions("test_ksp", "tab", token_range_size=int((2 ** 64) / 1000),
                                                     target_token_range_size=None))
        tkns_p = [i[1] for i in tkns_p]
        self.check_all(tkns_p, 1, 1000)

    def test_using_size_estimates(self):
        for i in range(100000):
            config.session.execute("INSERT INTO test_ksp.tab(k,v) values(%s,%s)", [i, i])
        test_config.ccm_cluster.flush()
        test_config.ccm_cluster.compact()

        obj = SimpleObj("test_ksp.tab")
        partitioner = Partitioner(obj, "SIMPLE")
        config.splits_per_node = 1
        tkns_p = list(
            partitioner._tokens_partitions("test_ksp", "tab", token_range_size=None, target_token_range_size=64))
        self.assertEqual(len(tkns_p), 2)
        # self.check_all(tkns_p, 1, 1000)

    def check_all(self, tkns_p, split_per_node, expected_total_tkns):
        self.assertGreaterEqual(len(tkns_p), len(test_config.ccm_cluster.nodes) * split_per_node)
        self.assertGreaterEqual(sum(map(len, tkns_p)), expected_total_tkns)
        self.check_full_range([i for split_tokens in tkns_p for i in split_tokens])

        hosts = [self.checkToken(worker_partition) for worker_partition in tkns_p]
        self.assertEqual(len(set(hosts)), len(test_config.ccm_cluster.nodes))
        self.ensure_balance(tkns_p)

    def check_full_range(self, list_of_ranges):
        list_of_ranges.sort()
        start = list(map(lambda a: a[0], list_of_ranges))
        counts = list(filter(lambda size: size[1] > 1, map(lambda number: (number, start.count(number)), start)))
        self.assertEqual(0, len(counts), "duplicated starts")
        end = list(map(lambda a: a[0], list_of_ranges))
        counts = list(filter(lambda size: size[1] > 1, map(lambda number: (number, end.count(number)), end)))
        self.assertEqual(0, len(counts), "duplicated ends")

        first, last = list_of_ranges[0]
        self.assertEqual(-(2 ** 63), first, "first token should always be -2^63")
        for s, e in list_of_ranges[1:]:
            self.assertEqual(last, s, "broken range %d -> %d" % (last, s))
            last = e
        self.assertEqual((2 ** 63) - 1, last, "last token should always be (2^63)-1")

    def ensure_balance(self, tokens_per_split):
        from cassandra.metadata import Token
        tm = config.cluster.metadata.token_map
        node_loads = defaultdict(int)
        for split in tokens_per_split:
            for f, t in split:
                host = tm.get_replicas("test_ksp", Token(f))[0]
                node_loads[host] += t - f

        print(node_loads)
        node_loads = node_loads.values()
        avg_delta = sum(node_loads) // len(node_loads)
        max_delta = max(node_loads)
        min_delta = min(node_loads)
        self.assertLessEqual(max_delta, avg_delta * 2)
        self.assertGreaterEqual(min_delta, avg_delta / 2)

    def checkToken(self, tokens):
        # type: (List[Long]) -> Host
        from cassandra.metadata import Token
        tm = config.cluster.metadata.token_map

        # only the first token of each partition is not assigned correctly
        tokens = [(tok[0] + 1, tok[1]) for tok in tokens]

        hosts = set(map(lambda token: tm.get_replicas("test_ksp", token)[0],
                        map(lambda a: Token(a[0]), tokens)))
        self.assertEqual(len(hosts), 1, "A token range is local in 2 nodes")
        return list(hosts)[0]


class IStorageSplitLocalityTestVnodes(IStorageSplitLocalityTest):
    @classmethod
    def setUpClass(cls):
        from hfetch import disconnectCassandra
        disconnectCassandra()
        from .. import set_ccm_cluster
        test_config.ccm_cluster.clear()
        set_ccm_cluster()
        from .. import TEST_DEBUG
        try:
            test_config.ccm_cluster.populate(3, use_vnodes=True).start()
        except Exception as ex:
            if not TEST_DEBUG:
                raise ex

        import hfetch
        import hecuba
        import importlib
        importlib.reload(hfetch)
        import importlib
        importlib.reload(hecuba)
        super(IStorageSplitLocalityTest, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        from hfetch import disconnectCassandra
        disconnectCassandra()

        test_config.ccm_cluster.clear()
        from .. import set_up_default_cassandra
        set_up_default_cassandra()


if __name__ == '__main__':
    unittest.main()
