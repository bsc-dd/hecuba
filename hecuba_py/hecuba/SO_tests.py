import unittest

from hecuba import StorageObj
from datetime import time

class TestAttributes(StorageObj):
    attr1: time

class MyTestCase(unittest.TestCase):
    def test_something(self):
        a = TestAttributes("test1")
        a.attr1 = time(12, 10, 30)
        a = TestAttributes("test1")
        self.assertEqual(a.attr1, time(12, 10, 30))


if __name__ == '__main__':
    unittest.main()
