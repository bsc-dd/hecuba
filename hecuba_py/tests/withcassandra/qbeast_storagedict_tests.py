import random
from random import random
import time
import unittest

from hecuba import config, StorageDict


class TestIndexObj(StorageDict):
    '''
    @TypeSpec dict<<partid:int, time:double>, x:double,y:double,z:double>
    @Index_on x,y,z
    '''


class QbeastStorageDictTest(unittest.TestCase):

    def testSimple(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.indexed_dict")
        config.session.execute("DROP TABLE IF EXISTS my_app_qbeast.indexed_dict_indexed_dict_idx_d8tree")
        d = TestIndexObj("my_app.indexed_dict")
        for i in range(0, 30):
            d[i, i+1.0] = [i*0.1/9.0, i*0.2/9.0, i*0.3/9.0]

        time.sleep(2)

        tree_dict = config.session.execute("SELECT * FROM my_app_qbeast.indexed_dict_indexed_dict_idx_d8tree")
        res = [[(row.partid, row.time), (row.x, row.y, row.z)] for row in tree_dict]
        self.assertEqual(30, len(res))

    def testIterator(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.indexed_dict")
        config.session.execute("DROP TABLE IF EXISTS my_app_qbeast.indexed_dict_indexed_dict_idx_d8tree")
        d = TestIndexObj("my_app.indexed_dict")
        what_should_be = dict()
        for i in range(0, 30):
            what_should_be[i, i+1.0] = [i*0.1/9.0, i*0.2/9.0, i*0.3/9.0]
            d[i, i+1.0] = [i*0.1/9.0, i*0.2/9.0, i*0.3/9.0]

        it = d.iteritems()
        for k, v in it:
            self.assertEqual(what_should_be[k], list(v))

    def testFilter(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.indexed_dict")
        config.session.execute("DROP TABLE IF EXISTS my_app_qbeast.indexed_dict_indexed_dict_idx_d8tree")
        d = TestIndexObj("my_app.indexed_dict")
        what_should_be = dict()
        for i in range(0, 30):
            what_should_be[i, i + 1.0] = [i*0.1/9.0, i*0.2/9.0, i*0.3/9.0]
            d[i, i + 1.0] = [i*0.1/9.0, i*0.2/9.0, i*0.3/9.0]

        time.sleep(2)

        filtered = filter(lambda row: row.x > 0.02 and row.x < 0.25 and row.y > 0.26 and row.y < 0.45 and row.z > 0.58 and row.z < 0.9, d.iteritems())
        normal_filtered = python_filter(lambda row: row[1][0] > 0.02 and row[1][0] < 0.25 and row[1][1] > 0.26 and row[1][1] < 0.45 and row[1][2] > 0.58 and row[1][2] < 0.9, what_should_be.iteritems())

        filtered_list = [row for row in filtered]
        self.assertEqual(len(filtered_list), len(normal_filtered))
        for row in filtered_list:
            self.assertTrue((tuple(row.key), list(row.value)) in normal_filtered)

    def testBuildRemotely(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.indexed_dict")
        config.session.execute("DROP TABLE IF EXISTS my_app_qbeast.indexed_dict_indexed_dict_idx_d8tree")
        d = TestIndexObj("my_app.indexed_dict")
        what_should_be = dict()
        for i in range(0, 30):
            what_should_be[i, i + 1.0] = [i*0.1/9.0, i*0.2/9.0, i*0.3/9.0]
            d[i, i + 1.0] = [i*0.1/9.0, i*0.2/9.0, i*0.3/9.0]

        res = config.session.execute(
            'SELECT storage_id, primary_keys, columns, class_name, name, tokens, istorage_props,indexed_on ' +
            'FROM hecuba.istorage WHERE storage_id = %s', [d._storage_id])[0]

        self.assertEqual(res.storage_id, d._storage_id)
        self.assertEqual(res.class_name, TestIndexObj.__module__ + "." + TestIndexObj.__name__)

        rebuild = StorageDict.build_remotely(res._asdict())
        self.assertEqual(rebuild._ksp, 'my_app')
        self.assertEqual(rebuild._table, 'indexed_dict')

        self.assertEqual(len(what_should_be), len([row for row in rebuild.iteritems()]))
        for k, v in rebuild.iteritems():
            self.assertEqual(what_should_be[k], list(v))

    def test_precision(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.indexed_dict")
        config.session.execute("DROP TABLE IF EXISTS my_app_qbeast.indexed_dict_indexed_dict_idx_d8tree")
        d = TestIndexObj("my_app.indexed_dict")
        what_should_be = dict()
        for i in range(0, 30):
            what_should_be[i, i + 1.0] = [i * 0.1 / 9.0, i * 0.2 / 9.0, i * 0.3 / 9.0]
            d[i, i + 1.0] = [i * 0.1 / 9.0, i * 0.2 / 9.0, i * 0.3 / 9.0]

        time.sleep(2)

        filtered = filter(lambda row: row.x > 0.02 and row.x < 0.25 and row.y > 0.26 and row.y < 0.45 and row.z > 0.58 and row.z < 0.9 and random.random() < 1, d.iteritems())
        normal_filtered = python_filter(lambda row: row[1][0] > 0.02 and row[1][0] < 0.25 and row[1][1] > 0.26 and row[1][1] < 0.45 and row[1][2] > 0.58 and row[1][2] < 0.9, what_should_be.iteritems())

        filtered_list = [row for row in filtered]
        self.assertEqual(len(filtered_list), len(normal_filtered))
        for row in filtered_list:
            self.assertTrue((tuple(row.key), list(row.value)) in normal_filtered)

        filtered = filter(lambda row: row.x > 0.02 and row.x < 0.25 and row.y > 0.26 and row.y < 0.45 and row.z > 0.58 and row.z < 0.9 and random() < 1, d.iteritems())

        filtered_list = [row for row in filtered]
        self.assertEqual(len(filtered_list), len(normal_filtered))
        for row in filtered_list:
            self.assertTrue((tuple(row.key), list(row.value)) in normal_filtered)

    # def testSplit(self):
    #     config.session.execute("DROP TABLE IF EXISTS my_app.indexed_dict")
    #     config.session.execute("DROP TABLE IF EXISTS my_app_qbeast.indexed_dict_indexed_dict_idx_d8tree")
    #     d = TestIndexObj("my_app.indexed_dict")
    #     what_should_be = dict()
    #     for i in range(0, 30):
    #         what_should_be[i, i + 1.0] = [i*0.1/9.0, i*0.2/9.0, i*0.3/9.0]
    #         d[i, i + 1.0] = [i*0.1/9.0, i*0.2/9.0, i*0.3/9.0]
    #
    #     time.sleep(2)
    #
    #     i = 0
    #     filtered = []
    #     normal_filtered = python_filter(lambda row: row[1][0] > 0.02 and row[1][0] < 0.25 and row[1][1] > 0.26 and row[1][1] < 0.45 and row[1][2] > 0.58 and row[1][2] < 0.9, what_should_be.iteritems())
    #
    #     for partition in d.split():
    #         # aggregation of filtering on each partition should be equal to a filter on the whole object
    #         f = filter(lambda row: row.x > 0.02 and row.x < 0.25 and row.y > 0.26 and row.y < 0.45 and row.z > 0.58 and row.z < 0.9, partition.iteritems())
    #         for k, v in f:
    #             filtered.append((tuple(k), list(v)))
    #
    #         for k, v in partition.iteritems():
    #             # self.assertTrue((tuple(row.key), list(row.value)) in f2)
    #             self.assertEqual(what_should_be[k], list(v))
    #             i += 1
    #
    #     self.assertEqual(len(what_should_be), i)
    #     self.assertEqual(len(normal_filtered), len(filtered))
    #     for row in filtered:
    #         self.assertTrue(row in normal_filtered)


if __name__ == '__main__':
    unittest.main()
