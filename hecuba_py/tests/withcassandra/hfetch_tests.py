import unittest

from hecuba import StorageDict


class ConcurrentDict(StorageDict):
    '''
    @TypeSpec <<key:int>,value:int>
    '''


class HfetchTests(unittest.TestCase):
    def test_timestamped_writes(self):

        my_dict = ConcurrentDict("concurrent_dict")
        last_value = 1000
        for value in range(last_value):
            my_dict[0] = value

        del my_dict
        my_dict = ConcurrentDict("concurrent_dict")

        retrieved = my_dict[0]



        self.assertEqual(retrieved, last_value - 1)

