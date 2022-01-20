import time
import unittest
from hecuba import StorageDict, config


class SimpleDict(StorageDict):
    '''
    @TypeSpec dict<<key0:int>, val0:int>
    '''


class ComplexDict(StorageDict):
    '''
    @TypeSpec dict<<key0:str, key1:int>, val0:str, val1:int, val2:float, val3:bool>
    '''


class LambdaParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = config.execution_name
        config.execution_name = "LambdaParserTest".lower()

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

    def test_simple_filter(self):
        simple_dict = SimpleDict("test_simple_filter")
        res = filter(lambda x: x.key0 == 5, simple_dict.items())
        res = [i for i in res]
        self.assertEqual(0, len(res))
        simple_dict.delete_persistent()

    def test_greater(self):
        simple_dict = SimpleDict("test_greater")
        for i in range(0, 10):
            simple_dict[i] = i
        time.sleep(1)

        res = filter(lambda x: x.key0 > 5, simple_dict.items())
        res = [i for i in res]
        self.assertEqual(4, len(res))
        self.assertTrue((6, 6) in res)
        self.assertTrue((7, 7) in res)
        self.assertTrue((8, 8) in res)
        self.assertTrue((9, 9) in res)
        simple_dict.delete_persistent()

    def test_column_not_exist(self):
        simple_dict = SimpleDict("test_column_not_exist")

        def filter_nonexisting_key():
            return filter(lambda x: x.key1 == 5, simple_dict.items())

        self.assertRaises(Exception, filter_nonexisting_key)
        simple_dict.delete_persistent()

    def test_not_persistent_object(self):
        simple_dict = SimpleDict()
        for i in range(0, 10):
            simple_dict[i] = i

        res = filter(lambda x: x[0] > 5, simple_dict.items())
        res = [i for i in res]
        self.assertEqual(4, len(res))
        self.assertTrue((6, 6) in res)
        self.assertTrue((7, 7) in res)
        self.assertTrue((8, 8) in res)
        self.assertTrue((9, 9) in res)

    def test_filter_equal(self):
        simple_dict = SimpleDict("test_filter_equal")
        for i in range(0, 10):
            simple_dict[i] = i
        time.sleep(1)

        res = filter(lambda x: x.key0 == 5, simple_dict.items())
        res = [i for i in res]
        self.assertEqual(1, len(res))
        self.assertEqual((5, 5), res[0])
        simple_dict.delete_persistent()

    def test_filter_inside(self):
        simple_dict = SimpleDict("test_filter_inside")
        for i in range(0, 10):
            simple_dict[i] = i
        time.sleep(1)

        res = filter(lambda x: x.key0 in [1, 3], simple_dict.items())
        res = [i for i in res]
        self.assertEqual(2, len(res))
        self.assertTrue((1, 1) in res)
        self.assertTrue((3, 3) in res)
        simple_dict.delete_persistent()

    def test_different_columns(self):
        simple_dict = SimpleDict("test_different_columns")
        for i in range(0, 10):
            simple_dict[i] = i
        time.sleep(1)

        res = filter(lambda x: x.key0 in [1, 2, 3, 5, 6, 9] and x.val0 >= 0 and x.val0 <= 5, simple_dict.items())
        res = [i for i in res]
        self.assertEqual(4, len(res))
        self.assertTrue((1, 1) in res)
        self.assertTrue((2, 2) in res)
        self.assertTrue((3, 3) in res)
        self.assertTrue((5, 5) in res)
        simple_dict.delete_persistent()

    def test_complex_filter(self):
        complex_dict = ComplexDict("test_complex_filter")
        for i in range(0, 20):
            complex_dict[str(i), i] = [str(i), i, float(i), True]
        time.sleep(2)

        res = filter(lambda x: x.key0 in ["1", "2", "3", "4", "5"] and x.val1 >= 1 and x.val1 <= 5 and x.val2 >= 1.0 and x.val2 <= 4.0 and x.val3 == True, complex_dict.items())
        res = [tuple(i) for i in res]
        self.assertEqual(4, len(res))
        self.assertTrue((("1", 1), ("1", 1, 1.0, True)) in res)
        self.assertTrue((("2", 2), ("2", 2, 2.0, True)) in res)
        self.assertTrue((("3", 3), ("3", 3, 3.0, True)) in res)
        self.assertTrue((("4", 4), ("4", 4, 4.0, True)) in res)
        complex_dict.delete_persistent()

    def test_bad_type(self):
        simple_dict = SimpleDict("test_bad_type")
        for i in range(0, 10):
            simple_dict[i] = i
        time.sleep(1)

        def execute_bad_type():
            res = filter(lambda x: x.key0 == "1", simple_dict.items())

        self.assertRaises(Exception, execute_bad_type)
        simple_dict.delete_persistent()

    def test_several_operators(self):
        simple_dict = SimpleDict("test_several_operators")
        for i in range(0, 10):
            simple_dict[i] = i
        time.sleep(1)

        res = filter(lambda x: x.key0 < 5 and x.key0 >= 3, simple_dict.items())
        res = [i for i in res]
        self.assertEqual(2, len(res))
        self.assertTrue((3, 3) in res)
        self.assertTrue((4, 4) in res)
        simple_dict.delete_persistent()

    def test_reversed_operations(self):
        simple_dict = SimpleDict("test_reversed_operations")
        for i in range(0, 10):
            simple_dict[i] = i
        time.sleep(1)

        res = filter(lambda x: 5 > x.key0 and 3 <= x.key0, simple_dict.items())
        res = [i for i in res]
        self.assertEqual(2, len(res))
        self.assertTrue((3, 3) in res)
        self.assertTrue((4, 4) in res)
        simple_dict.delete_persistent()

    def test_non_hecuba_filter(self):
        l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        res = list(filter(lambda x: x >= 5, l))
        self.assertEqual(res, [5, 6, 7, 8, 9])

    def test_split_filter(self):
        simple_dict = SimpleDict("test_split_filter")
        what_should_be = dict()
        for i in range(0, 10):
            what_should_be[i] = i
            simple_dict[i] = i
        time.sleep(1)

        filtered = []
        normal_filtered = list(python_filter(lambda x: x[0] > 3, simple_dict.items()))

        i = 0
        for partition in simple_dict.split():
            # aggregation of filtering on each partition should be equal to a filter on the whole object
            res = filter(lambda x: x.key0 > 3, partition.items())
            for row in res:
                filtered.append(row)

            for k, v in partition.items():
                # self.assertTrue((tuple(row.key), list(row.value)) in f2)
                self.assertEqual(what_should_be[k], v)
                i += 1

        self.assertEqual(len(what_should_be), i)
        self.assertEqual(len(filtered), len(normal_filtered))
        for row in filtered:
            self.assertTrue(row in normal_filtered)
        simple_dict.delete_persistent()


if __name__ == "__main__":
    unittest.main()
