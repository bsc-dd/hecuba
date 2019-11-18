import os
import time
import unittest
from random import randint

from hecuba import config, StorageDict


class MyDict(StorageDict):
    '''
    @TypeSpec dict<<key0:int>, val0:str>
    '''


set_start_time = """UPDATE hecuba.partitioning
                    SET start_time = %s
                    WHERE storage_id = %s"""

set_time = """UPDATE hecuba.partitioning
              SET start_time = %s, end_time = %s
              WHERE storage_id = %s"""

set_end_time = """UPDATE hecuba.partitioning
                  SET end_time = %s
                  WHERE storage_id = %s"""


class PartitionerTest(unittest.TestCase):

    def computeItems(self, SDict):
        counter = 0
        for _ in SDict.keys():
            counter = counter + 1
        return counter

    def test_simple(self):
        config.splits_per_node = 32
        config.session.execute("DROP TABLE IF EXISTS my_app.mydict")
        d = MyDict("my_app.mydict")
        nitems = 10000
        for i in range(0, nitems):
            d[i] = "RandomText" + str(i)

        time.sleep(2)
        # assert all the data has been written
        self.assertEqual(len(list(d.keys())), nitems)

        acc = 0
        nsplits = 0
        config.partition_strategy = "SIMPLE"
        for partition in d.split():
            nsplits += 1
            acc += self.computeItems(partition)

        nodes_number = 2
        print("number of splits: %s, best is %s" % (nsplits, config.splits_per_node * nodes_number))
        self.assertEqual(nitems, acc)

    def test_dynamic_simple(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mydict")
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        d = MyDict("my_app.mydict")
        nitems = 10000
        for i in range(0, nitems):
            d[i] = "RandomText" + str(i)

        time.sleep(2)
        # assert all the data has been written
        self.assertEqual(len(list(d.keys())), nitems)

        acc = 0
        os.environ["NODES_NUMBER"] = "3"
        config.partition_strategy = "DYNAMIC"
        granularity = [32, 45]
        times = [(0, 80), (0, 5)]
        nsplits = 0
        for partition in d.split():
            if nsplits <= 1:
                # this will be done by the compss api
                config.session.execute(set_time, [times[nsplits][0], times[nsplits][1], partition.storage_id])
            nsplits += 1

            acc += self.computeItems(partition)

        print("number of splits: %s, best is %s" % (nsplits, 45))
        self.assertEqual(nitems, acc)
        self.assertEqual(config.splits_per_node, 45 // 2)

    def test_dynamic_simple_other(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mydict")
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        d = MyDict("my_app.mydict")
        nitems = 10000
        for i in range(0, nitems):
            d[i] = "RandomText" + str(i)

        time.sleep(2)
        # assert all the data has been written
        self.assertEqual(len(list(d.keys())), nitems)

        acc = 0
        os.environ["NODES_NUMBER"] = "3"
        config.partition_strategy = "DYNAMIC"
        granularity = [32, 45]
        times = [(0, 10), (0, 80)]
        nsplits = 0
        for partition in d.split():
            if nsplits <= 1:
                # this will be done by the compss api
                config.session.execute(set_time, [times[nsplits][0], times[nsplits][1], partition.storage_id])
            nsplits += 1

            acc += self.computeItems(partition)

        print("number of splits: %s, best is %s" % (nsplits, 32))
        self.assertEqual(nitems, acc)
        self.assertEqual(config.splits_per_node, 32 // 2)

    def test_dynamic_different_nodes(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mydict")
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        d = MyDict("my_app.mydict")
        nitems = 10000
        for i in range(0, nitems):
            d[i] = "RandomText" + str(i)

        time.sleep(2)
        # assert all the data has been written
        self.assertEqual(len(list(d.keys())), nitems)

        acc = 0
        os.environ["NODES_NUMBER"] = "5"
        config.partition_strategy = "DYNAMIC"
        granularity = [32, 45, 64, 90]
        times = [(0, 1000), (0, 1000), (0, 20), (0, 1000)]
        nsplits = 0
        for partition in d.split():
            if nsplits <= 3:
                # this will be done by the compss api
                config.session.execute(set_time, [times[nsplits][0], times[nsplits][1], partition.storage_id])
            nsplits += 1

            acc += self.computeItems(partition)

        print("number of splits: %s, best is %s\n" % (nsplits, 64))
        self.assertEqual(nitems, acc)
        self.assertEqual(config.splits_per_node, 64 // 2)

    def test_dynamic_best_without_finishing(self):
        """
        Test if the best granularity is set without finishing all the initial granularities tasks.
        This happens when all the unfinished tasks are worse than the best granularity with at least one finished task
        """
        config.session.execute("DROP TABLE IF EXISTS my_app.mydict")
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        d = MyDict("my_app.mydict")
        nitems = 10000
        for i in range(0, nitems):
            d[i] = "RandomText" + str(i)

        time.sleep(2)
        # assert all the data has been written
        self.assertEqual(len(list(d.keys())), nitems)

        acc = 0
        os.environ["NODES_NUMBER"] = "3"
        config.partition_strategy = "DYNAMIC"
        granularity = [32, 45]
        times = [(0, 80), (0, 40)]
        nsplits = 0
        for partition in d.split():
            # pretending that task with gran=32 is taking a lot of time
            if nsplits == 0:
                # this will be done by the compss api
                config.session.execute(set_start_time, [0, partition.storage_id])
            elif nsplits == 1:
                config.session.execute(set_time, [times[nsplits][0], times[nsplits][1], partition.storage_id])
            else:
                self.assertEqual(config.splits_per_node, 45 // 2)

            nsplits += 1
            acc += self.computeItems(partition)

        print("number of splits: %s, best is %s" % (nsplits, 45))
        self.assertEqual(nitems, acc)
        self.assertEqual(config.splits_per_node, 45 // 2)

    def test_dynamic_best_idle_nodes(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mydict")
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        d = MyDict("my_app.mydict")
        nitems = 10000
        for i in range(0, nitems):
            d[i] = "RandomText" + str(i)

        time.sleep(2)
        # assert all the data has been written
        self.assertEqual(len(list(d.keys())), nitems)

        acc = 0
        os.environ["NODES_NUMBER"] = "3"
        config.partition_strategy = "DYNAMIC"
        granularity = [32, 45]
        times = [(0, 120), (0, 60)]
        nsplits = 0
        for partition in d.split():
            # pretending that task with gran=32 is taking a lot of time
            if nsplits == 0:
                id_partition0 = partition.storage_id
                # this will be done by the compss api
                config.session.execute(set_start_time, [time.time(), partition.storage_id])
            elif nsplits == 1:
                config.session.execute(set_time, [times[nsplits][0], times[nsplits][1], partition.storage_id])
            elif nsplits == 5:
                config.session.execute(set_end_time, [time.time() + 150, id_partition0])
            elif 1 < nsplits < 5:
                start = randint(0, 200)
                config.session.execute(set_time, [start, start + 60, partition.storage_id])

            if nsplits > 1:
                self.assertEqual(config.splits_per_node, 45 // 2)

            nsplits += 1
            acc += self.computeItems(partition)

        print("number of splits: %s, best is %s" % (nsplits, 45))
        self.assertEqual(nitems, acc)
        self.assertEqual(config.splits_per_node, 45 // 2)

    def test_dynamic_idle_nodes_new_best(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mydict")
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        d = MyDict("my_app.mydict")
        nitems = 10000
        for i in range(0, nitems):
            d[i] = "RandomText" + str(i)

        time.sleep(2)
        # assert all the data has been written
        self.assertEqual(len(list(d.keys())), nitems)

        acc = 0
        os.environ["NODES_NUMBER"] = "3"
        config.partition_strategy = "DYNAMIC"
        granularity = [32, 45]
        times = [(0, 80), (0, 60)]
        nsplits = 0
        for partition in d.split():
            if nsplits == 0:
                id_partition0 = partition.storage_id
                # this will be done by the compss api
                # time.time() to avoid choosing gran=64 when task with gran=32 taking a lot of time
                # dynamic partitioning mode will use time.time() to check how much is taking
                config.session.execute(set_start_time, [time.time(), partition.storage_id])
            elif nsplits == 1:
                config.session.execute(set_time, [times[nsplits][0], times[nsplits][1], partition.storage_id])
            elif nsplits == 5:
                last_time = config.session.execute("""SELECT start_time FROM hecuba.partitioning
                                                      WHERE storage_id = %s""" % id_partition0)[0][0]
                config.session.execute(set_end_time, [last_time + 80, id_partition0])
            else:
                start = randint(0, 200)
                config.session.execute(set_time, [start, start + 60, partition.storage_id])

            if 5 >= nsplits >= 2:
                self.assertEqual(config.splits_per_node, 45 // 2)
            elif nsplits > 5:
                self.assertEqual(config.splits_per_node, 32 // 2)
            nsplits += 1
            acc += self.computeItems(partition)

        print("number of splits: %s, best is %s" % (nsplits, 32))
        self.assertEqual(nitems, acc)
        self.assertEqual(config.splits_per_node, 32 // 2)

    def test_check_nodes_not_set(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.mydict")
        d = MyDict("my_app.mydict")

        def raise_exception():
            return [partition for partition in d.split()]

        config.partition_strategy = "DYNAMIC"

        if "NODES_NUMBER" in os.environ:
            del os.environ["NODES_NUMBER"]
        if "PYCOMPSS_NODES" in os.environ:
            del os.environ["PYCOMPSS_NODES"]

        self.assertRaises(RuntimeError, raise_exception)


if __name__ == "__main__":
    unittest.main()
