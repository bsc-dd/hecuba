from hecuba.hset import StorageSet
from hecuba import config
import unittest


class SetStr(StorageSet):
    '''
    @TypeSpec str
    '''

class SetInt(StorageSet):
    '''
    @TypeSpec int
    '''

class SetTuple(StorageSet):
    '''
    @TypeSpec <int, str, str>
    '''

class SetTest(unittest.TestCase):
    # TESTS WITH STRINGS

    def testAddRemoveStr(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.settests")
        setTests = SetStr("pruebas2.settests")
        setTests.add("1")
        setTests.add("2")
        setTests.remove("1")

        self.assertTrue("2" in setTests)
        self.assertFalse("2" not in setTests)
        self.assertTrue("1" not in setTests)

    def testIterationStr(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.settests")
        setTests = SetStr("pruebas2.settests")

        for i in range(0, 10):
            setTests.add(str(i))

        for i in range(0, 10):
            self.assertTrue(str(i) in setTests)

    def testLenStr(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.settests")
        setTests = SetStr("pruebas2.settests")

        for i in range(0, 10):
            setTests.add(str(i))

        self.assertEqual(10, len(setTests))
        self.assertNotEqual(5, len(setTests))

    def testUnionStr(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set1")
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set2")
        set1 = SetStr("pruebas2.set1")
        set2 = SetStr("pruebas2.set2")

        for i in range(0, 10):
            set1.add(str(i))
        for i in range(10, 20):
            set2.add(str(i))

        set3 = set1.union(set2)

        for i in range(0, 20):
            self.assertTrue(str(i) in set3)

        self.assertEqual(20, len(set3))

    def testIntersectionStr(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set1")
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set2")
        set1 = SetStr("pruebas2.set1")
        set2 = SetStr("pruebas2.set2")

        for i in range(0, 10):
            set1.add(str(i))
        for i in range(5, 15):
            set2.add(str(i))

        set3 = set1.intersection(set2)

        for i in range(5, 10):
            self.assertTrue(str(i) in set3)

        self.assertEqual(5, len(set3))

    def testDifferenceStr(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set1")
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set2")
        set1 = SetStr("pruebas2.set1")
        set2 = SetStr("pruebas2.set2")

        for i in range(0, 10):
            set1.add(str(i))
        for i in range(5, 15):
            set2.add(str(i))

        set3 = set1.difference(set2)

        for i in range(0, 5):
            self.assertTrue(str(i) in set3)

        self.assertEqual(5, len(set3))

    def testClearStr(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set1")
        set1 = SetStr("pruebas2.set1")
        for i in range(0, 10):
            set1.add(str(i))

        set1.clear()

        for i in range(0, 10):
            self.assertFalse(str(i) in set1)

        self.assertEqual(0, len(set1))

    # TESTS WITH INTS

    def testAddRemoveInt(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.settests2")
        setTests = SetInt("pruebas2.settests2")
        setTests.add(1)
        setTests.add(2)
        setTests.remove(1)

        self.assertTrue(2 in setTests)
        self.assertFalse(2 not in setTests)
        self.assertTrue(1 not in setTests)

    def testIterationInt(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.settests2")
        setTests = SetInt("pruebas2.settests2")

        for i in range(0, 10):
            setTests.add(i)

        for i in range(0, 10):
            self.assertTrue(i in setTests)

    def testLenInt(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.settests2")
        setTests = SetInt("pruebas2.settests2")

        for i in range(0, 10):
            setTests.add(i)

        self.assertEqual(10, len(setTests))
        self.assertNotEqual(5, len(setTests))

    def testUnionInt(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set3")
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set4")
        set1 = SetInt("pruebas2.set3")
        set2 = SetInt("pruebas2.set4")

        for i in range(0, 10):
            set1.add(i)
        for i in range(10, 20):
            set2.add(i)

        set3 = set1.union(set2)

        for i in range(0, 20):
            self.assertTrue(i in set3)

        self.assertEqual(20, len(set3))

    def testIntersectionInt(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set3")
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set4")
        set1 = SetInt("pruebas2.set3")
        set2 = SetInt("pruebas2.set4")

        for i in range(0, 10):
            set1.add(i)
        for i in range(5, 15):
            set2.add(i)

        set3 = set1.intersection(set2)

        for i in range(5, 10):
            self.assertTrue(i in set3)

        self.assertEqual(5, len(set3))

    def testDifferenceInt(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set3")
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set4")
        set1 = SetInt("pruebas2.set3")
        set2 = SetInt("pruebas2.set4")

        for i in range(0, 10):
            set1.add(i)
        for i in range(5, 15):
            set2.add(i)

        set3 = set1.difference(set2)

        for i in range(0, 5):
            self.assertTrue(i in set3)

        self.assertEqual(5, len(set3))

    def testClearInt(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set3")
        set1 = SetInt("pruebas2.set3")
        for i in range(0, 10):
            set1.add(i)

        set1.clear()
        for i in range(0, 10):
            self.assertFalse(i in set1)

        self.assertEqual(0, len(set1))

    def testMakePersistent(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set3")
        set3 = SetInt("pruebas2.set3")
        set3.delete_persistent()

        set3 = SetInt()
        for i in range(0, 10):
            set3.add(i)

        set3.remove(0)
        set3.remove(1)

        set3.make_persistent("pruebas2.set3")
        for i in range(2, 10):
            self.assertTrue(i in set3)

        self.assertFalse(0 in set3)
        self.assertFalse(1 in set3)

    def testTupleAdd(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.setTuple1")
        setTuple = SetTuple("pruebas2.setTuple1")
        for i in range(0, 10):
            if i%2 == 0:
                setTuple.add((i, "Mesa", "Marron"))
            else:
                setTuple.add((i, "Silla", "Negra"))

        self.assertTrue((0, "Mesa", "Marron") in setTuple)

        nmesas = nsillas = 0
        for value in setTuple:
            if "Mesa" in value:
                nmesas += 1
            else:
                if "Silla" in value:
                    nsillas += 1
        self.assertEqual(5, nmesas)
        self.assertEqual(5, nsillas)

    def testTupleRemove(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.setTuple1")
        setTuple = SetTuple("pruebas2.setTuple1")
        for i in range(0, 10):
            if i % 2 == 0:
                setTuple.add((i, "Mesa", "Marron"))
            else:
                setTuple.add((i, "Silla", "Negra"))

        for i in range(5, 10):
            if i % 2 == 0:
                setTuple.remove((i, "Mesa", "Marron"))
            else:
                setTuple.remove((i, "Silla", "Negra"))
        nmesas = nsillas = 0
        for value in setTuple:
            if "Mesa" in value:
                nmesas += 1
            else:
                if "Silla" in value:
                    nsillas += 1
        self.assertEqual(3, nmesas)
        self.assertEqual(2, nsillas)

    def testTupleUnion(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.setTuple1")
        setTuple1 = SetTuple("pruebas2.setTuple1")
        for i in range(0, 10):
            if i % 2 == 0:
                setTuple1.add((i, "Mesa", "Marron"))
            else:
                setTuple1.add((i, "Silla", "Negra"))
        config.session.execute("DROP TABLE IF EXISTS pruebas2.setTuple2")
        setTuple2 = SetTuple("pruebas2.setTuple2")
        setTuple2.add((0, "Paraguas", "Rojo"))
        setTuple2.add((1, "Paraguas", "Verde"))

        set3 = setTuple1.union(setTuple2)

        for i in range(0, 10):
            if i % 2 == 0:
                self.assertTrue((i, "Mesa", "Marron") in set3)
            else:
                self.assertTrue((i, "Silla", "Negra") in set3)

        self.assertTrue((0, "Paraguas", "Rojo") in set3)
        self.assertTrue((1, "Paraguas", "Verde") in set3)

    def testTupleIntersection(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.setTuple1")
        setTuple1 = SetTuple("pruebas2.setTuple1")
        for i in range(0, 10):
            if i % 2 == 0:
                setTuple1.add((i, "Mesa", "Marron"))
            else:
                setTuple1.add((i, "Silla", "Negra"))
        config.session.execute("DROP TABLE IF EXISTS pruebas2.setTuple2")
        setTuple2 = SetTuple("pruebas2.setTuple2")
        setTuple2.add((0, "Mesa", "Marron"))
        setTuple2.add((1, "Silla", "Negra"))

        set3 = setTuple1.intersection(setTuple2)

        self.assertTrue((0, "Mesa", "Marron") in set3)
        self.assertTrue((1, "Silla", "Negra") in set3)
        for i in range(2, 10):
            if i % 2 == 0:
                self.assertFalse((i, "Mesa", "Marron") in set3)
            else:
                self.assertFalse((i, "Silla", "Negra") in set3)

    def testTupleDifference(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.setTuple1")
        setTuple1 = SetTuple("pruebas2.setTuple1")
        for i in range(0, 10):
            if i % 2 == 0:
                setTuple1.add((i, "Mesa", "Marron"))
            else:
                setTuple1.add((i, "Silla", "Negra"))
        config.session.execute("DROP TABLE IF EXISTS pruebas2.setTuple2")
        setTuple2 = SetTuple("pruebas2.setTuple2")
        setTuple2.add((0, "Mesa", "Marron"))
        setTuple2.add((1, "Silla", "Negra"))

        set3 = setTuple1.difference(setTuple2)

        self.assertFalse((0, "Mesa", "Marron") in set3)
        self.assertFalse((1, "Silla", "Negra") in set3)
        for i in range(2, 10):
            if i % 2 == 0:
                self.assertTrue((i, "Mesa", "Marron") in set3)
            else:
                self.assertTrue((i, "Silla", "Negra") in set3)

    def testTupleMakePersistent(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.setTuple1")
        setTuple = SetTuple("pruebas2.setTuple1")
        setTuple.delete_persistent()

        setTuple = SetTuple()
        for i in range(0, 10):
            if i % 2 == 0:
                setTuple.add((i, "Mesa", "Marron"))
            else:
                setTuple.add((i, "Silla", "Negra"))

        setTuple.remove((0, "Mesa", "Marron"))
        setTuple.remove((1, "Silla", "Negra"))

        setTuple.make_persistent("pruebas2.setTuple1")
        for i in range(2, 10):
            if i % 2 == 0:
                self.assertTrue((i, "Mesa", "Marron") in setTuple)
            else:
                self.assertTrue((i, "Silla", "Negra") in setTuple)

        self.assertFalse((0, "Mesa", "Marron") in setTuple)
        self.assertFalse((1, "Silla", "Negra") in setTuple)

    def testUnionWithSet(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set3")
        set1 = SetInt("pruebas2.set3")
        for i in range(0, 10):
            set1.add(i)
        set2 = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}

        set3 = set1.union(set2)

        for i in range(0, 20):
            self.assertTrue(i in set3)

        self.assertEqual(20, len(set3))

    def testIntersectionWithSet(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set3")
        set1 = SetInt("pruebas2.set3")
        for i in range(0, 10):
            set1.add(i)
        set2 = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

        set3 = set1.intersection(set2)

        for i in range(5, 10):
            self.assertTrue(i in set3)

        self.assertEqual(5, len(set3))

    def testDifferenceWithSet(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set3")
        set1 = SetInt("pruebas2.set3")
        for i in range(0, 10):
            set1.add(i)
        set2 = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

        set3 = set1.difference(set2)

        for i in range(0, 5):
            self.assertTrue(i in set3)

        self.assertEqual(5, len(set3))

    def testPostUnionPersistence(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set3")
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set4")
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set5")
        set1 = SetInt("pruebas2.set3")
        set2 = SetInt("pruebas2.set4")

        for i in range(0, 10):
            set1.add(i)
        for i in range(10, 20):
            set2.add(i)

        set3 = set1.union(set2)
        set3.make_persistent("pruebas2.set5")
        set4 = SetInt("pruebas2.set5")

        for i in range(0, 20):
            self.assertTrue(i in set4)

        self.assertEqual(20, len(set4))

    def testUnionTwoNoPersistent(self):
        set1 = SetInt()
        set2 = SetInt()
        for i in range(0, 10):
            set1.add(i)
        for i in range(10, 20):
            set2.add(i)

        set3 = set1.union(set2)

        for i in range(0, 20):
            self.assertTrue(i in set3)

        self.assertEqual(20, len(set3))

    def testIntersectionTwoNoPersistent(self):
        set1 = SetInt()
        set2 = SetInt()
        for i in range(0, 10):
            set1.add(i)
        for i in range(5, 15):
            set2.add(i)

        set3 = set1.intersection(set2)

        for i in range(5, 10):
            self.assertTrue(i in set3)

        self.assertEqual(5, len(set3))

    def testDifferenceTwoNoPersistent(self):
        set1 = SetInt()
        set2 = SetInt()
        for i in range(0, 10):
            set1.add(i)
        for i in range(5, 15):
            set2.add(i)

        set3 = set1.difference(set2)

        for i in range(0, 5):
            self.assertTrue(i in set3)

        self.assertEqual(5, len(set3))

    def testStopPersistentButInMemory(self):
        config.session.execute("DROP TABLE IF EXISTS pruebas2.set1")
        set1 = SetInt("pruebas2.set1")
        set1.add(2)
        set1.add(3)
        set1.stop_persistent()
        self.assertTrue(2 in set1)
        self.assertTrue(3 in set1)


if __name__ == '__main__':
    unittest.main()
