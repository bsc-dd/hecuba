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
        setTests = SetStr("pruebas2.settests")
        setTests.clear()
        setTests.add("1")
        setTests.add("2")
        setTests.remove("1")

        self.assertTrue("2" in setTests)
        self.assertFalse("2" not in setTests)
        self.assertTrue("1" not in setTests)

    def testIterationStr(self):
        setTests = SetStr("pruebas2.settests")
        setTests.clear()

        for i in range(0, 10):
            setTests.add(str(i))

        for i in range(0, 10):
            self.assertTrue(str(i) in setTests)

    def testLenStr(self):
        setTests = SetStr("pruebas2.settests")
        setTests.clear()

        for i in range(0, 10):
            setTests.add(str(i))

        self.assertEqual(10, len(setTests))
        self.assertNotEqual(5, len(setTests))

    def testUnionStr(self):
        set1 = SetStr("pruebas2.set1")
        set2 = SetStr("pruebas2.set2")
        set1.clear()
        set2.clear()

        for i in range(0, 10):
            set1.add(str(i))
        for i in range(10, 20):
            set2.add(str(i))

        set1.union(set2)

        for i in range(0, 20):
            self.assertTrue(str(i) in set1)

        self.assertEqual(20, len(set1))

    def testIntersectionStr(self):
        set1 = SetStr("pruebas2.set1")
        set2 = SetStr("pruebas2.set2")
        set1.clear()
        set2.clear()
        for i in range(0, 10):
            set1.add(str(i))
        for i in range(5, 15):
            set2.add(str(i))

        set1.intersection(set2)

        for i in range(5, 10):
            self.assertTrue(str(i) in set1)

        self.assertEqual(5, len(set1))

    def testDifferenceStr(self):
        set1 = SetStr("pruebas2.set1")
        set2 = SetStr("pruebas2.set2")
        set1.clear()
        set2.clear()
        for i in range(0, 10):
            set1.add(str(i))
        for i in range(5, 15):
            set2.add(str(i))

        set1.difference(set2)

        for i in range(0, 5):
            self.assertTrue(str(i) in set1)

        self.assertEqual(5, len(set1))

    def testClearStr(self):
        set1 = SetStr("pruebas2.set1")
        for i in range(0, 10):
            set1.add(str(i))

        set1.clear()

        for i in range(0, 10):
            self.assertFalse(str(i) in set1)

        self.assertEqual(0, len(set1))

    # TESTS WITH INTS

    def testAddRemoveInt(self):
        setTests = SetInt("pruebas2.settests2")
        setTests.clear()
        setTests.add(1)
        setTests.add(2)
        setTests.remove(1)

        self.assertTrue(2 in setTests)
        self.assertFalse(2 not in setTests)
        self.assertTrue(1 not in setTests)

    def testIterationInt(self):
        setTests = SetInt("pruebas2.settests2")
        setTests.clear()

        for i in range(0, 10):
            setTests.add(i)

        for i in range(0, 10):
            self.assertTrue(i in setTests)

    def testLenInt(self):
        setTests = SetInt("pruebas2.settests2")
        setTests.clear()

        for i in range(0, 10):
            setTests.add(i)

        self.assertEqual(10, len(setTests))
        self.assertNotEqual(5, len(setTests))

    def testUnionInt(self):
        set1 = SetInt("pruebas2.set3")
        set2 = SetInt("pruebas2.set4")
        set1.clear()
        set2.clear()

        for i in range(0, 10):
            set1.add(i)
        for i in range(10, 20):
            set2.add(i)

        set1.union(set2)

        for i in range(0, 20):
            self.assertTrue(i in set1)

        self.assertEqual(20, len(set1))

    def testIntersectionInt(self):
        set1 = SetInt("pruebas2.set3")
        set2 = SetInt("pruebas2.set4")
        set1.clear()
        set2.clear()
        for i in range(0, 10):
            set1.add(i)
        for i in range(5, 15):
            set2.add(i)

        set1.intersection(set2)

        for i in range(5, 10):
            self.assertTrue(i in set1)

        self.assertEqual(5, len(set1))

    def testDifferenceInt(self):
        set1 = SetStr("pruebas2.set3")
        set2 = SetStr("pruebas2.set4")
        set1.clear()
        set2.clear()
        for i in range(0, 10):
            set1.add(i)
        for i in range(5, 15):
            set2.add(i)

        set1.difference(set2)

        for i in range(0, 5):
            self.assertTrue(i in set1)

        self.assertEqual(5, len(set1))

    def testClearInt(self):
        set1 = SetInt("pruebas2.set3")
        for i in range(0, 10):
            set1.add(i)

        set1.clear()

        for i in range(0, 10):
            self.assertFalse(i in set1)

        self.assertEqual(0, len(set1))

    def testMakePersistent(self):
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
        setTuple = SetTuple("pruebas2.setTuple1")
        setTuple.clear()
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
        setTuple = SetTuple("pruebas2.setTuple1")
        setTuple.clear()
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
        setTuple1 = SetTuple("pruebas2.setTuple1")
        setTuple1.clear()
        for i in range(0, 10):
            if i % 2 == 0:
                setTuple1.add((i, "Mesa", "Marron"))
            else:
                setTuple1.add((i, "Silla", "Negra"))
        setTuple2 = SetTuple("pruebas2.setTuple2")
        setTuple2.clear()
        setTuple2.add((0, "Paraguas", "Rojo"))
        setTuple2.add((1, "Paraguas", "Verde"))

        setTuple1.union(setTuple2)

        for i in range(0, 10):
            if i % 2 == 0:
                self.assertTrue((i, "Mesa", "Marron") in setTuple1)
            else:
                self.assertTrue((i, "Silla", "Negra") in setTuple1)

        self.assertTrue((0, "Paraguas", "Rojo") in setTuple1)
        self.assertTrue((1, "Paraguas", "Verde") in setTuple1)

    def testTupleIntersection(self):
        setTuple1 = SetTuple("pruebas2.setTuple1")
        setTuple1.clear()
        for i in range(0, 10):
            if i % 2 == 0:
                setTuple1.add((i, "Mesa", "Marron"))
            else:
                setTuple1.add((i, "Silla", "Negra"))
        setTuple2 = SetTuple("pruebas2.setTuple2")
        setTuple2.clear()
        setTuple2.add((0, "Mesa", "Marron"))
        setTuple2.add((1, "Silla", "Negra"))

        setTuple1.intersection(setTuple2)

        self.assertTrue((0, "Mesa", "Marron") in setTuple1)
        self.assertTrue((1, "Silla", "Negra") in setTuple1)
        for i in range(2, 10):
            if i % 2 == 0:
                self.assertFalse((i, "Mesa", "Marron") in setTuple1)
            else:
                self.assertFalse((i, "Silla", "Negra") in setTuple1)

    def testTupleDifference(self):
        setTuple1 = SetTuple("pruebas2.setTuple1")
        setTuple1.clear()
        for i in range(0, 10):
            if i % 2 == 0:
                setTuple1.add((i, "Mesa", "Marron"))
            else:
                setTuple1.add((i, "Silla", "Negra"))
        setTuple2 = SetTuple("pruebas2.setTuple2")
        setTuple2.clear()
        setTuple2.add((0, "Mesa", "Marron"))
        setTuple2.add((1, "Silla", "Negra"))

        setTuple1.difference(setTuple2)

        self.assertFalse((0, "Mesa", "Marron") in setTuple1)
        self.assertFalse((1, "Silla", "Negra") in setTuple1)
        for i in range(2, 10):
            if i % 2 == 0:
                self.assertTrue((i, "Mesa", "Marron") in setTuple1)
            else:
                self.assertTrue((i, "Silla", "Negra") in setTuple1)

    def testTupleMakePersistent(self):
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
        set1 = SetInt("pruebas2.set3")
        set1.clear()
        for i in range(0, 10):
            set1.add(i)
        set2 = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}

        set1.union(set2)

        for i in range(0, 20):
            self.assertTrue(i in set1)

        self.assertEqual(20, len(set1))

    def testIntersectionWithSet(self):
        set1 = SetInt("pruebas2.set3")
        set1.clear()
        for i in range(0, 10):
            set1.add(i)
        set2 = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

        set1.intersection(set2)

        for i in range(5, 10):
            self.assertTrue(i in set1)

        self.assertEqual(5, len(set1))

    def testDifferenceWithSet(self):
        set1 = SetStr("pruebas2.set3")
        set1.clear()
        for i in range(0, 10):
            set1.add(i)
        set2 = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

        set1.difference(set2)

        for i in range(0, 5):
            self.assertTrue(i in set1)

        self.assertEqual(5, len(set1))


if __name__ == '__main__':
    unittest.main()
