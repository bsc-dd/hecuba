from hecuba_py.src.hset import StorageSet
import unittest


class SetStr(StorageSet):
    '''
    @TypeSpec str
    '''

class SetInt(StorageSet):
    '''
    @TypeSpec int
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

    def testClearInt(self):
        set1 = SetInt("pruebas2.set3")
        for i in range(0, 10):
            set1.add(i)

        set1.clear()

        for i in range(0, 10):
            self.assertFalse(i in set1)

        self.assertEqual(0, len(set1))

if __name__ == '__main__':
    unittest.main()
