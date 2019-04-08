import cassandra
import unittest

from hecuba import config, StorageObj, StorageDict
from ..app.words import Words
import uuid
import time


class MyStorageDict(StorageObj):
    '''
    @ClassField table1 dict<<key:int>, val:int>
    '''


class mydict(StorageDict):
    '''
    @ClassField table2 dict<<key0:int>, val0:tests.withcassandra.storage_name_tests.myobj2>
    '''


class myobj2(StorageObj):
    '''
    @ClassField attr1 int
    @ClassField attr2 str
    '''


class StorageDictTest(unittest.TestCase):
    def test_attribute_double_assignation(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.somename")
        dic = MyStorageDict("some")

        dic.table1[0] = [1]
        self.assertEqual(dic.table1[0], 1)

        dic.table1[0] = [2]
        self.assertEqual(dic.table1[0], 2)

        dic2 = MyStorageDict("some")
        self.assertEqual(dic2.table1[0], 2)

    def test_attribute_double_assignation_so(self):
        config.session.execute("DROP TABLE IF EXISTS my_app.somename")
        dic = mydict()

        so1 = myobj2()
        so1.attr1 = 1
        so1.attr2 = "a"

        so2 = myobj2()
        so2.attr1 = 2
        so2.attr2 = "b"

        dic[0] = so1
        dic[0] = so2

        self.assertEqual(dic[0], so2)
if __name__ == '__main__':
    unittest.main()
