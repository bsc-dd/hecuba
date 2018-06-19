import unittest


class StorageTests(unittest.TestCase):
    def setUp(self):
        from hecuba import Config
        Config.reset(mock_cassandra=False)

    def test_getByID_block(self):
        from storage.api import getByID
        from class_definitions import Words
        # ki = KeyIter('testspace', 'tt', 'app.words.Words', 'fake-id', ['position'])
        SO = Words('so')
        b = SO.split().next()
        new_block = getByID(b.getID())
        self.assertEqual(b.getID(), new_block.getID())
        self.assertEqual(b, new_block)

    def test_getByID_storage_obj(self):
        from storage.api import getByID
        from class_definitions import Words
        b = Words('testspace.tt')
        new_block = getByID(b.getID())
        self.assertEqual(b, new_block)

if __name__ == '__main__':
    unittest.main()
