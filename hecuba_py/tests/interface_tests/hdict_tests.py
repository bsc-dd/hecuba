import unittest

from hecuba import StorageDict
from typing_extensions import TypedDict


class MyTestCase(unittest.TestCase):
    def test_dict_str_int(self):
        b = StorageDict('movie', TypedDict('Movie', {'name': str, 'year': int}))
        b['year'] = 2000
        b['name'] = 'ala'
        a = StorageDict('movie')
        print(a['year'])
        print(a['name'])




if __name__ == '__main__':
    unittest.main()
