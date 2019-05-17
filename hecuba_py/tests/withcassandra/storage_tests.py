import unittest

from ..app.words import Words
from hecuba import Config, config
from storage.api import getByID


class StorageTests(unittest.TestCase):

    def setUp(self):
        Config.reset(mock_cassandra=False)

    def test_getByID_block(self):
        # ki = KeyIter('testspace', 'tt', 'app.words.Words', 'fake-id', ['position'])
        SO = Words('so')
        b = next(SO.split())
        new_block = getByID(b.getID())
        self.assertEqual(b.getID(), new_block.getID())
        self.assertEqual(b, new_block)

    def test_getByID_storage_obj(self):
        b = Words('testspace.tt')
        new_block = getByID(b.getID())
        self.assertEqual(b, new_block)

if __name__ == '__main__':
    unittest.main()
