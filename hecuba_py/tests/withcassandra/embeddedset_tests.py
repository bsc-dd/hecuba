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
    def setUpClass(cls):
        cls.old = config.execution_name
        config.execution_name = "EmbeddedSetTest".lower()
    @classmethod
    def tearDownClass(cls):
        #config.session.execute("DROP KEYSPACE IF EXISTS {}".format(config.execution_name))
        config.execution_name = cls.old

    # Create a new keyspace per test
    def setUp(self):
        self.current_ksp = config.execution_name
        pass

    def tearDown(self):
        pass

    def testAddRemove(self):
        d = DictSet2(self.current_ksp+".testAddRemove")
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

    def testDoNotCollide(self):
        d = DictSet2(self.current_ksp+".testDoNotCollide")
        d["1", 1] = {"1"}
        d["2", 2] = {"1", "2", "3"}

        self.assertTrue("1" in d["1", 1])
        for i in range(1, 4):
            self.assertTrue(str(i) in d["2", 2])

        del d
        d2 = DictSet2(self.current_ksp+".testDoNotCollide")
        self.assertTrue("1" in d2["1", 1])
        for i in range(1, 4):
            self.assertTrue(str(i) in d2["2", 2])

    def testDoNotCollideEmptySet(self):
        d = DictSet2(self.current_ksp+".testDoNotCollideEmptySet")
        d["1", 1] = {"1", "2", "3"}
        d["2", 2] = {"4", "5", "6"}

        del d
        d2 = DictSet2(self.current_ksp+".testDoNotCollideEmptySet")

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

    def testAddRemove2(self):
        d = DictSet4(self.current_ksp+".testAddRemove2")
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

    def testAddRemoveInt(self):
        d = DictSet3(self.current_ksp+".testAddRemoveInt")
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

    def testIter(self):
        d = DictSet2(self.current_ksp+".testIter")

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

    def testAddRemoveTuple(self):
        d = DictSet(self.current_ksp+".testAddRemoveTuple")
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

    def testIterTuple(self):
        d = DictSet(self.current_ksp+".testIterTuple")

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

    def testUnion(self):
        d1 = DictSet2(self.current_ksp+".testUnion1")
        d2 = DictSet2(self.current_ksp+".testUnion2")

        d1["0", 0] = set()
        d2["1", 1] = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
        for i in range(10, 20):
            d2["1", 1].add(str(i))

        d3 = DictSet2(self.current_ksp+".testUnion3")
        d3["2", 2] = d1["0", 0].union(d2["1", 1])
        time.sleep(3)
        for i in range(0, 20):
            self.assertTrue(str(i) in d3["2", 2])
        self.assertEqual(20, len(d3["2", 2]))

    def testIntersection(self):
        d1 = DictSet2(self.current_ksp+".testIntersection1")
        d2 = DictSet2(self.current_ksp+".testIntersection2")

        d1["0", 0] = set()
        d2["1", 1] = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
            if i < 5:
                d2["1", 1].add(str(i))

        d3 = DictSet2(self.current_ksp+".testIntersection3")
        d3["2", 2] = d1["0", 0].intersection(d2["1", 1])
        for i in range(0, 5):
            self.assertTrue(str(i) in d3["2", 2])
        self.assertEqual(5, len(d3["2", 2]))

    def testDifference(self):
        d1 = DictSet2("testDifference1")
        d2 = DictSet2("testDifference2")

        d1["0", 0] = set()
        d2["1", 1] = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
            if i < 5:
                d2["1", 1].add(str(i))

        d3 = DictSet2("testDifference3")
        d3["2", 2] = d1["0", 0].difference(d2["1", 1])
        for i in range(5, 10):
            self.assertTrue(str(i) in d3["2", 2])
        self.assertEqual(5, len(d3["2", 2]))

    def testUnionWithTuples(self):
        d1 = DictSet("testUnionWithTuples1")
        d2 = DictSet("testUnionWithTuples2")

        d1["0", 0] = set()
        d2["1", 1] = set()
        for i in range(0, 10):
            d1["0", 0].add((i, i))
        for i in range(10, 20):
            d2["1", 1].add((i, i))

        d3 = DictSet("testUnionWithTuples3")
        d3["2", 2] = d1["0", 0].union(d2["1", 1])
        time.sleep(3)
        for i in range(0, 20):
            self.assertTrue((i, i) in d3["2", 2])
        self.assertEqual(20, len(d3["2", 2]))

    def testIntersectionWithTuples(self):
        d1 = DictSet("testIntersectionWithTuples1")
        d2 = DictSet("testIntersectionWithTuples2")

        d1["0", 0] = set()
        d2["1", 1] = set()
        for i in range(0, 10):
            d1["0", 0].add((i, i))
            if i < 5:
                d2["1", 1].add((i, i))

        d3 = DictSet("testIntersectionWithTuples3")
        d3["2", 2] = d1["0", 0].intersection(d2["1", 1])
        for i in range(0, 5):
            self.assertTrue((i, i) in d3["2", 2])
        self.assertEqual(5, len(d3["2", 2]))

    def testDifferenceWithTuples(self):
        d1 = DictSet("testDifferenceWithTuples1")
        d2 = DictSet("testDifferenceWithTuples2")

        d1["0", 0] = set()
        d2["1", 1] = set()
        for i in range(0, 10):
            d1["0", 0].add((i, i))
            if i < 5:
                d2["1", 1].add((i, i))

        d3 = DictSet("testDifferenceWithTuples3")
        d3["2", 2] = d1["0", 0].difference(d2["1", 1])
        for i in range(5, 10):
            self.assertTrue((i, i) in d3["2", 2])
        self.assertEqual(5, len(d3["2", 2]))

    def testUnionWithSet(self):
        d1 = DictSet2("testUnionWithSet1")

        d1["0", 0] = set()
        s1 = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
            s1.add(str(i + 10))

        d3 = DictSet2("testUnionWithSet3")
        d3["2", 2] = d1["0", 0].union(s1)
        time.sleep(3)
        for i in range(0, 20):
            self.assertTrue(str(i) in d3["2", 2])
        self.assertEqual(20, len(d3["2", 2]))

    def testIntersectionWithSet(self):
        d1 = DictSet2("testIntersectionWithSet1")

        d1["0", 0] = set()
        s1 = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
            if i < 5:
                s1.add(str(i))

        d3 = DictSet2("testIntersectionWithSet3")
        d3["2", 2] = d1["0", 0].intersection(s1)
        for i in range(0, 5):
            self.assertTrue(str(i) in d3["2", 2])
        self.assertEqual(5, len(d3["2", 2]))

    def testDifferenceWithSet(self):
        d1 = DictSet2("testDifferenceWithSet1")

        d1["0", 0] = set()
        s1 = set()
        for i in range(0, 10):
            d1["0", 0].add(str(i))
            if i < 5:
                s1.add(str(i))

        d3 = DictSet2("testDifferenceWithSet3")
        d3["2", 2] = d1["0", 0].difference(s1)
        for i in range(5, 10):
            self.assertTrue(str(i) in d3["2", 2])
        self.assertEqual(5, len(d3["2", 2]))

    def testIterKeys(self):
        d1 = DictSet3("testIterKeys1")

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

    def testItems(self):
        d1 = DictSet3("testItems")

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

    def testIterValues(self):
        d1 = DictSet3("testIterValues")

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

    def testItemsWithTuples(self):
        d1 = DictSet("testItemsWithTuples")

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

    def testUpdate(self):
        d1 = DictSet2("testUpdate1")
        d2 = DictSet2("testUpdate2")

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

    def testIsSubset(self):
        d1 = DictSet2("testIsSubset1")
        d2 = DictSet2("testIsSubset2")

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

    def testIsSuperset(self):
        d1 = DictSet2("testIsSuperset1")
        d2 = DictSet2("testIsSuperset2")

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

    def testIsSubsetWithSet(self):
        d1 = DictSet2("testIsSubsetWithSet")

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

    def testIsSupersetWithSet(self):
        d1 = DictSet2("testIsSupersetWithSet")

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

    def testRemoveKeyError(self):
        d1 = DictSet2("testRemoveKeyError")

        d1["0", 0] = {"0", "1", "2"}
        d1["0", 0].remove("0")
        self.assertRaises(Exception, d1["0", 0].remove, "3")
        self.assertTrue("1" in d1["0", 0])
        self.assertTrue("2" in d1["0", 0])
        self.assertEqual(False, d1["0", 0].__contains__("0"))

    def testDiscard(self):
        d1 = DictSet2("testDiscard")

        d1["0", 0] = {"0", "1", "2"}
        d1["0", 0].discard("0")
        d1["0", 0].discard("3")
        self.assertTrue("1" in d1["0", 0])
        self.assertTrue("2" in d1["0", 0])
        self.assertEqual(False, d1["0", 0].__contains__("0"))

    def testClear(self):
        d1 = DictSet3("testClear")

        d1["0", 0] = {1, 2}
        time.sleep(1)
        self.assertEqual(2, len(d1["0", 0]))
        d1["0", 0].clear()
        self.assertEqual(0, len(d1["0", 0]))

    def testSplit(self):
        d = DictSet3("testSplit")
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

    def testBuildRemotely(self):
        d = DictSet("testbuildremotely")
        for i in range(0, 10):
            d[str(i), i] = {(0, 0), (1, 1), (2, 2)}

        self.assertTrue(d._table is not None)
        self.assertEqual(self.current_ksp, d._ksp)
        d.sync() #guarantee that data is on disk before reading it again

        res = config.session.execute(
            'SELECT storage_id, primary_keys, columns, class_name, name, tokens, istorage_props,indexed_on ' +
            'FROM hecuba.istorage WHERE storage_id = %s', [d.storage_id])[0]

        self.assertEqual(res.storage_id, d.storage_id)
        self.assertEqual(res.class_name, DictSet.__module__ + "." + DictSet.__name__)
        self.assertEqual(res.name, self.current_ksp+'.testbuildremotely')

        rebuild = build_remotely(res._asdict())
        self.assertEqual(rebuild._built_remotely, True)
        self.assertEqual('testbuildremotely', rebuild._table)
        self.assertEqual(self.current_ksp, rebuild._ksp)
        self.assertEqual(res.storage_id, rebuild.storage_id)

        self.assertEqual(d._is_persistent, rebuild._is_persistent)

        for i in range(0, 10):
            # rebuild[str(i), i] does not return data, we must iterate over it
            mys = {k for k in rebuild[str(i), i]}
            for val in {(0,0), (1, 1), (2, 2)}:
                self.assertTrue( val in mys )


if __name__ == '__main__':
    unittest.main()
