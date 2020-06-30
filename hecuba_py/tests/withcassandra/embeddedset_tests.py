import time
import unittest

from hecuba import StorageDict
from hecuba import config
from hecuba.IStorage \
    import build_remotely


class DictSet(StorageDict):
    '''
    @TypeSpec dict<<k1:str, k2:int>, s1:set<int, int>>
    '''


class DictSet2(StorageDict):
    '''
    @TypeSpec dict<<k1:str, k2:int>, s1:set<str>>
    '''


class DictSet3(StorageDict):
    '''
    @TypeSpec dict<<k1:str, k2:int>, s1:set<int>>
    '''


class DictSet4(StorageDict):
    '''
    @TypeSpec dict<<k1:str>, s1:set<int>>
    '''


class EmbeddedSetTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        config.session.execute("DROP KEYSPACE IF EXISTS pruebas0;")

    def tearDown(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas0.dictset;")
        config.session.execute("DROP TABLE IF EXISTS pruebas0.dictset1;")
        config.session.execute("DROP TABLE IF EXISTS pruebas0.dictset2;")
        config.session.execute("DROP TABLE IF EXISTS pruebas0.dictset3;")

    def testAddRemove(self):
        d = DictSet2("pruebas0.dictset")
        d["1", 1] = {"1"}
        d["2", 2] = {"1", "2", "3"}

        self.assertTrue("1" in d["1", 1])
        for i in range(1, 4):
            self.assertTrue(str(i) in d["2", 2])

        d["2", 2].remove("2")
        self.assertEqual(False, d["2", 2].__contains__("2"))

        d["2", 2].add("2")
        self.assertTrue("2" in d["2", 2])
        self.assertEqual(1, len(d["1", 1]))
        self.assertEqual(3, len(d["2", 2]))
        #clean up
        d.delete_persistent()

    def testDoNotCollide(self):
        d = DictSet2("pruebas0.dictset")
        d["1", 1] = {"1"}
        d["2", 2] = {"1", "2", "3"}

        self.assertTrue("1" in d["1", 1])
        for i in range(1, 4):
            self.assertTrue(str(i) in d["2", 2])

        del d
        d2 = DictSet2("pruebas0.dictset")
        self.assertTrue("1" in d2["1", 1])
        for i in range(1, 4):
            self.assertTrue(str(i) in d2["2", 2])
        #clean up
        d2.delete_persistent()

    def testDoNotCollideEmptySet(self):
        d = DictSet2("pruebas0.dictset")
        d["1", 1] = {"1", "2", "3"}
        d["2", 2] = {"4", "5", "6"}

        del d
        d2 = DictSet2("pruebas0.dictset")

        d2["1", 1].add("4")
        d2["2", 2].add("7")
        self.assertTrue("4" in d2["1", 1])
        self.assertTrue("7" in d2["2", 2])

        d2["1", 1] = {"1"}
        d2["2", 2] = {"1", "2", "3"}
        self.assertTrue("1" in d2["1", 1])
        for i in range(1, 4):
            self.assertTrue(str(i) in d2["2", 2])

        self.assertEqual(len(d2["1", 1]), 1)
        self.assertEqual(len(d2["2", 2]), 3)
        #clean up
        d2.delete_persistent()

    def testAddRemove2(self):
        d = DictSet4("pruebas0.dictset")
        d["10"] = {1}
        d["20"] = {1, 2, 3}

        self.assertTrue(1 in d["10"])
        for i in range(1, 4):
            self.assertTrue(i in d["20"])

        d["20"].remove(2)
        self.assertEqual(False, d["20"].__contains__("2"))

        d["20"].add(2)
        self.assertTrue(2 in d["20"])
        self.assertEqual(1, len(d["10"]))
        self.assertEqual(3, len(d["20"]))
        #clean up
        d.delete_persistent()

    def testAddRemoveInt(self):
        d = DictSet3("pruebas0.dictset")
        d["1", 1] = {1}
        d["2", 2] = {1, 2, 3}

        self.assertTrue(1 in d["1", 1])
        for i in range(1, 4):
            self.assertTrue(i in d["2", 2])

        d["2", 2].remove(2)

        self.assertEqual(False, d["2", 2].__contains__(2))

        d["2", 2].add(2)
        self.assertTrue(2 in d["2", 2])
        self.assertEqual(1, len(d["1", 1]))
        self.assertEqual(3, len(d["2", 2]))
        #clean up
        d.delete_persistent()

    def testIter(self):
        d = DictSet2("pruebas0.dictset")

        d["2", 2] = set()
        for i in range(0, 10):
            d["2", 2].add(str(i))
        time.sleep(1)
        l = []
        for value in d["2", 2]:
            l.append(value)
        for i in range(0, 10):
            self.assertTrue(str(i) in l)

        self.assertEqual(len(l), len(d["2", 2]))
        #clean up
        d.delete_persistent()

    def testAddRemoveTuple(self):
        d = DictSet("pruebas0.dictset")
        d["1", 1] = {(1, 1)}
        d["2", 2] = {(1, 1), (2, 2), (3, 3)}

        self.assertTrue((1, 1) in d["1", 1])
        for i in range(1, 4):
            self.assertTrue((i, i) in d["2", 2])

        d["2", 2].remove((2, 2))
        self.assertEqual(False, d["2", 2].__contains__((2, 2)))

        d["2", 2].add((2, 2))
        self.assertTrue((2, 2) in d["2", 2])
        self.assertEqual(1, len(d["1", 1]))
        self.assertEqual(3, len(d["2", 2]))
        #clean up
        d.delete_persistent()

    def testIterTuple(self):
        d = DictSet("pruebas0.dictset")

        d["2", 2] = set()
        for i in range(0, 10):
            d["2", 2].add((i, i))
        time.sleep(1)
        l = []
        for value in d["2", 2]:
            l.append(value)
        for i in range(0, 10):
            self.assertTrue((i, i) in l)

        self.assertEqual(len(l), len(d["2", 2]))
        #clean up
        d.delete_persistent()

    def testUnion(self):
        d1 = DictSet2("pruebas0.dictset1")
        d2 = DictSet2("pruebas0.dictset2")

        d1["0", 0] = set()
        d2["1", 1] = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
        for i in range(10, 20):
            d2["1", 1].add(str(i))

        d3 = DictSet2("pruebas0.dictset3")
        d3["2", 2] = d1["0", 0].union(d2["1", 1])
        time.sleep(3)
        for i in range(0, 20):
            self.assertTrue(str(i) in d3["2", 2])
        self.assertEqual(20, len(d3["2", 2]))
        #clean up
        d1.delete_persistent()
        d2.delete_persistent()
        d3.delete_persistent()

    def testIntersection(self):
        d1 = DictSet2("pruebas0.dictset1")
        d2 = DictSet2("pruebas0.dictset2")

        d1["0", 0] = set()
        d2["1", 1] = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
            if i < 5:
                d2["1", 1].add(str(i))

        d3 = DictSet2("pruebas0.dictset3")
        d3["2", 2] = d1["0", 0].intersection(d2["1", 1])
        for i in range(0, 5):
            self.assertTrue(str(i) in d3["2", 2])
        self.assertEqual(5, len(d3["2", 2]))
        #clean up
        d1.delete_persistent()
        d2.delete_persistent()
        d3.delete_persistent()

    def testDifference(self):
        d1 = DictSet2("pruebas0.dictset1")
        d2 = DictSet2("pruebas0.dictset2")

        d1["0", 0] = set()
        d2["1", 1] = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
            if i < 5:
                d2["1", 1].add(str(i))

        d3 = DictSet2("pruebas0.dictset3")
        d3["2", 2] = d1["0", 0].difference(d2["1", 1])
        for i in range(5, 10):
            self.assertTrue(str(i) in d3["2", 2])
        self.assertEqual(5, len(d3["2", 2]))
        #clean up
        d1.delete_persistent()
        d2.delete_persistent()
        d3.delete_persistent()

    def testUnionWithTuples(self):
        d1 = DictSet("pruebas0.dictset1")
        d2 = DictSet("pruebas0.dictset2")

        d1["0", 0] = set()
        d2["1", 1] = set()
        for i in range(0, 10):
            d1["0", 0].add((i, i))
        for i in range(10, 20):
            d2["1", 1].add((i, i))

        d3 = DictSet("pruebas0.dictset3")
        d3["2", 2] = d1["0", 0].union(d2["1", 1])
        time.sleep(3)
        for i in range(0, 20):
            self.assertTrue((i, i) in d3["2", 2])
        self.assertEqual(20, len(d3["2", 2]))
        #clean up
        d1.delete_persistent()
        d2.delete_persistent()
        d3.delete_persistent()

    def testIntersectionWithTuples(self):
        d1 = DictSet("pruebas0.dictset1")
        d2 = DictSet("pruebas0.dictset2")

        d1["0", 0] = set()
        d2["1", 1] = set()
        for i in range(0, 10):
            d1["0", 0].add((i, i))
            if i < 5:
                d2["1", 1].add((i, i))

        d3 = DictSet("pruebas0.dictset3")
        d3["2", 2] = d1["0", 0].intersection(d2["1", 1])
        for i in range(0, 5):
            self.assertTrue((i, i) in d3["2", 2])
        self.assertEqual(5, len(d3["2", 2]))
        #clean up
        d1.delete_persistent()
        d2.delete_persistent()
        d3.delete_persistent()

    def testDifferenceWithTuples(self):
        d1 = DictSet("pruebas0.dictset1")
        d2 = DictSet("pruebas0.dictset2")

        d1["0", 0] = set()
        d2["1", 1] = set()
        for i in range(0, 10):
            d1["0", 0].add((i, i))
            if i < 5:
                d2["1", 1].add((i, i))

        d3 = DictSet("pruebas0.dictset3")
        d3["2", 2] = d1["0", 0].difference(d2["1", 1])
        for i in range(5, 10):
            self.assertTrue((i, i) in d3["2", 2])
        self.assertEqual(5, len(d3["2", 2]))
        #clean up
        d1.delete_persistent()
        d2.delete_persistent()
        d3.delete_persistent()

    def testUnionWithSet(self):
        d1 = DictSet2("pruebas0.dictset1")

        d1["0", 0] = set()
        s1 = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
            s1.add(str(i + 10))

        d3 = DictSet2("pruebas0.dictset3")
        d3["2", 2] = d1["0", 0].union(s1)
        time.sleep(3)
        for i in range(0, 20):
            self.assertTrue(str(i) in d3["2", 2])
        self.assertEqual(20, len(d3["2", 2]))
        #clean up
        d1.delete_persistent()
        d3.delete_persistent()

    def testIntersectionWithSet(self):
        d1 = DictSet2("pruebas0.dictset1")

        d1["0", 0] = set()
        s1 = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
            if i < 5:
                s1.add(str(i))

        d3 = DictSet2("pruebas0.dictset3")
        d3["2", 2] = d1["0", 0].intersection(s1)
        for i in range(0, 5):
            self.assertTrue(str(i) in d3["2", 2])
        self.assertEqual(5, len(d3["2", 2]))
        #clean up
        d1.delete_persistent()
        d3.delete_persistent()

    def testDifferenceWithSet(self):
        d1 = DictSet2("pruebas0.dictset1")

        d1["0", 0] = set()
        s1 = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
            if i < 5:
                s1.add(str(i))

        d3 = DictSet2("pruebas0.dictset3")
        d3["2", 2] = d1["0", 0].difference(s1)
        for i in range(5, 10):
            self.assertTrue(str(i) in d3["2", 2])
        self.assertEqual(5, len(d3["2", 2]))
        #clean up
        d1.delete_persistent()
        d3.delete_persistent()

    def testIterKeys(self):
        d1 = DictSet3("pruebas0.dictset1")

        d1["0", 0] = {1, 2, 3, 4}
        d1["0", 1] = {1, 2, 3, 4}
        d1["1", 0] = {1, 2, 3, 4}
        d1["1", 1] = {1, 2, 3, 4}
        time.sleep(1)
        l = list()
        for keys in d1.keys():
            l.append(keys)

        self.assertTrue(("0", 0) in l)
        self.assertTrue(("0", 1) in l)
        self.assertTrue(("1", 0) in l)
        self.assertTrue(("1", 1) in l)
        self.assertEqual(4, len(l))
        #clean up
        d1.delete_persistent()

    def testItems(self):
        d1 = DictSet3("pruebas0.dictset1")

        d1["0", 0] = {1, 2, 3, 4}
        d1["0", 1] = {1, 2, 3, 4}
        d1["1", 0] = {1, 2, 3, 4}
        d1["1", 1] = {1, 2, 3, 4}
        time.sleep(1)
        d = dict()
        for keys, value in d1.items():
            d[keys] = value
        self.assertTrue(("0", 0) in d)
        self.assertTrue(("0", 1) in d)
        self.assertTrue(("1", 0) in d)
        self.assertTrue(("1", 1) in d)
        self.assertEqual(4, len(d))
        self.assertEqual({1, 2, 3, 4}, d["0", 0])
        self.assertEqual({1, 2, 3, 4}, d["0", 1])
        self.assertEqual({1, 2, 3, 4}, d["1", 0])
        self.assertEqual({1, 2, 3, 4}, d["1", 1])
        #clean up
        d1.delete_persistent()

    def testIterValues(self):
        d1 = DictSet3("pruebas0.dictset1")

        d1["0", 0] = {1, 2, 3, 4}
        d1["0", 1] = {5, 6, 7, 8}
        d1["1", 0] = {8, 7, 6, 5}
        d1["1", 1] = {4, 3, 2, 1}
        time.sleep(1)
        l = list()
        for value in d1.values():
            l.append(value)

        self.assertEqual(4, len(l))
        self.assertTrue({1, 2, 3, 4} in l)
        self.assertTrue({5, 6, 7, 8} in l)
        self.assertTrue({8, 7, 6, 5} in l)
        self.assertTrue({4, 3, 2, 1} in l)
        #clean up
        d1.delete_persistent()

    def testItemsWithTuples(self):
        d1 = DictSet("pruebas0.dictset1")

        d1["key0", 15] = {(0, 1), (2, 3), (4, 5)}
        d1["key1", 30] = {(10, 11), (12, 13), (14, 15)}
        time.sleep(1)
        d = dict()
        for keys, value in d1.items():
            d[keys] = value
        self.assertTrue(("key0", 15) in d)
        self.assertTrue(("key1", 30) in d)
        self.assertEqual(2, len(d))
        self.assertEqual({(0, 1), (2, 3), (4, 5)}, d["key0", 15])
        self.assertEqual({(10, 11), (12, 13), (14, 15)}, d["key1", 30])
        #clean up
        d1.delete_persistent()

    def testUpdate(self):
        d1 = DictSet2("pruebas0.dictset1")
        d2 = DictSet2("pruebas0.dictset2")

        d1["0", 0] = set()
        d2["1", 1] = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
        for i in range(5, 20):
            d2["1", 1].add(str(i))

        time.sleep(2)
        d1["0", 0].update(d2["1", 1])
        time.sleep(3)
        for i in range(0, 20):
            self.assertTrue(str(i) in d1["0", 0])
        self.assertEqual(20, len(d1["0", 0]))
        #clean up
        d1.delete_persistent()
        d2.delete_persistent()

    def testIsSubset(self):
        d1 = DictSet2("pruebas0.dictset1")
        d2 = DictSet2("pruebas0.dictset2")

        d1["0", 0] = set()
        d2["1", 1] = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
            d2["1", 1].add(str(i))
        for i in range(10, 20):
            d2["1", 1].add(str(i))

        b = d1["0", 0].issubset(d2["1", 1])
        self.assertTrue(b)

        b = d2["1", 1].issubset(d1["0", 0])
        self.assertFalse(b)
        #clean up
        d1.delete_persistent()
        d2.delete_persistent()

    def testIsSuperset(self):
        d1 = DictSet2("pruebas0.dictset1")
        d2 = DictSet2("pruebas0.dictset2")

        d1["0", 0] = set()
        d2["1", 1] = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
            d2["1", 1].add(str(i))
        for i in range(10, 20):
            d2["1", 1].add(str(i))

        time.sleep(2)
        b = d2["1", 1].issuperset(d1["0", 0])
        self.assertTrue(b)

        b = d1["0", 0].issuperset(d2["1", 1])
        self.assertFalse(b)
        #clean up
        d1.delete_persistent()
        d2.delete_persistent()

    def testIsSubsetWithSet(self):
        d1 = DictSet2("pruebas0.dictset1")

        d1["0", 0] = set()
        s = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
            s.add(str(i))
        for i in range(10, 20):
            s.add(str(i))

        b = d1["0", 0].issubset(s)
        self.assertTrue(b)

        b = s.issubset(d1["0", 0])
        self.assertFalse(b)
        #clean up
        d1.delete_persistent()

    def testIsSupersetWithSet(self):
        d1 = DictSet2("pruebas0.dictset1")

        d1["0", 0] = set()
        s = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
            s.add(str(i))
        for i in range(10, 20):
            s.add(str(i))

        b = s.issuperset(d1["0", 0])
        self.assertTrue(b)

        b = d1["0", 0].issuperset(s)
        self.assertFalse(b)
        #clean up
        d1.delete_persistent()

    def testRemoveKeyError(self):
        d1 = DictSet2("pruebas0.dictset1")

        d1["0", 0] = {"0", "1", "2"}
        d1["0", 0].remove("0")
        self.assertRaises(Exception, d1["0", 0].remove, "3")
        self.assertTrue("1" in d1["0", 0])
        self.assertTrue("2" in d1["0", 0])
        self.assertEqual(False, d1["0", 0].__contains__("0"))
        #clean up
        d1.delete_persistent()

    def testDiscard(self):
        d1 = DictSet2("pruebas0.dictset1")

        d1["0", 0] = {"0", "1", "2"}
        d1["0", 0].discard("0")
        d1["0", 0].discard("3")
        self.assertTrue("1" in d1["0", 0])
        self.assertTrue("2" in d1["0", 0])
        self.assertEqual(False, d1["0", 0].__contains__("0"))
        #clean up
        d1.delete_persistent()

    def testClear(self):
        d1 = DictSet3("pruebas0.dictset1")

        d1["0", 0] = {1, 2}
        time.sleep(1)
        self.assertEqual(2, len(d1["0", 0]))
        d1["0", 0].clear()
        self.assertEqual(0, len(d1["0", 0]))
        #clean up
        d1.delete_persistent()

    def testSplit(self):
        d = DictSet3("pruebas0.dictset1")
        for i in range(0, 10):
            for j in range(0, 3):
                d[str(i), j] = {0, 1, 2, 3, 4, 5}

        d2 = dict()
        for partition in d.split():
            for ((key0, key1), val) in partition.items():
                d2[key0, key1] = val

        for i in range(0, 10):
            for j in range(0, 3):
                self.assertEqual(d2[str(i), j], {0, 1, 2, 3, 4, 5})
        #clean up
        d.delete_persistent()

    def testBuildRemotely(self):
        d = DictSet("pruebas0.dictset1")
        for i in range(0, 10):
            d[str(i), i] = {(0, 0), (1, 1), (2, 2)}

        self.assertEqual('dictset1', d._table)
        self.assertEqual('pruebas0', d._ksp)

        res = config.session.execute(
            'SELECT storage_id, primary_keys, columns, class_name, name, tokens, istorage_props,indexed_on ' +
            'FROM hecuba.istorage WHERE storage_id = %s', [d.storage_id])[0]

        self.assertEqual(res.storage_id, d.storage_id)
        self.assertEqual(res.class_name, DictSet.__module__ + "." + DictSet.__name__)
        self.assertEqual(res.name, 'pruebas0.dictset1')

        rebuild = build_remotely(res._asdict())
        self.assertEqual(rebuild._built_remotely, True)
        self.assertEqual('dictset1', rebuild._table)
        self.assertEqual('pruebas0', rebuild._ksp)
        self.assertEqual(res.storage_id, rebuild.storage_id)

        self.assertEqual(d._is_persistent, rebuild._is_persistent)

        for i in range(0, 10):
            # rebuild[str(i), i] does not return data, we must iterate over it
            self.assertEqual({i for i in rebuild[str(i), i]}, {(0, 0), (1, 1), (2, 2)})
        #clean up
        d.delete_persistent()


if __name__ == '__main__':
    unittest.main()
