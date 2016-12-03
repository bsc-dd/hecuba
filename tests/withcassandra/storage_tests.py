import unittest

from app.words import Words
from hecuba import Config, config
from hecuba.iter import KeyIter
from storage.api import getByID


class StorageTests(unittest.TestCase):

    def setUp(self):
        Config.reset(mock_cassandra=False)

    def test_getByID_block(self):
        config.number_of_blocks = 32
        ki = KeyIter('testspace', 'tt', 'app.words.Words', 'fake-id', ['position'])
        b = ki.next()
        new_block = getByID(b.getID())
        self.assertEqual(b.storageobj.getID(), new_block.storageobj.getID())
        self.assertEqual(b, new_block)

    def test_getByID_storage_obj(self):
        b = Words('testspace.tt')
        self.assertRegexpMatches(b.getID(), '.*_1')
        new_block = getByID(b.getID())
        self.assertEqual(b, new_block)

if __name__ == '__main__':
    unittest.main()
