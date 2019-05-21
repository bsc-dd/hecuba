import unittest

from hecuba import config, StorageDict


class ConcurrentDict(StorageDict):
    '''
    @TypeSpec <<key:int>,value:int>
    '''


class HfetchTests(unittest.TestCase):
    def test_timestamped_writes(self):
        previous_cfg = config.timestamped_writes
        config.timestamped_writes = "True"

        my_dict = ConcurrentDict("concurrent_dict")
        last_value = 1000
        for value in xrange(last_value):
            my_dict[0] = value

        del my_dict
        my_dict = ConcurrentDict("concurrent_dict")

        retrieved = my_dict[0]

        config.timestamped_writes = previous_cfg

        self.assertEqual(retrieved, last_value - 1)

