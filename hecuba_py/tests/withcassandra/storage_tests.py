import unittest

from ..app.words import Words
from storage.api import getByID


class StorageTests(unittest.TestCase):
    def test_getByID_block(self):
        # ki = KeyIter('testspace', 'tt', 'app.words.Words', 'fake-id', ['position'])
        from hecuba import config
        config.session.execute("DROP TABLE IF EXISTS my_app.so")
        config.session.execute("DROP TABLE IF EXISTS my_app.so_0")
        config.session.execute("DROP TABLE IF EXISTS my_app.so_1")
        SO = Words('so')
        b = next(SO.split())
        new_block = getByID(b.get_id())
        self.assertEqual(b.get_id(), new_block.get_id())
        self.assertEqual(b, new_block)

    def test_getByID_storage_obj(self):
        b = Words('testspace.tt')
        new_block = getByID(b.get_id())
        self.assertEqual(b, new_block)


if __name__ == '__main__':
    unittest.main()
