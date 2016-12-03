import unittest

from hecuba.iter import Block
from storage.api import getByID, start_task,end_task
import uuid

from setup import config


class ApiTest(unittest.TestCase):
    @staticmethod
    def setUpClass():
        config.reset(mock_cassandra=False)

    def test_end_task(self):
        self.fail('to be done')
    def test_start_task(self):
        self.fail('to be done')

    def test_getByID_block(self):
        bid = str(uuid.uuid1())
        b = Block(bid, 'localhost', ['pk1'], [1, 3], 'hecuba.storageobj.StorageObj')
        self.assertEqual(b.getID(), bid)
        newBlock = getByID(b.getID())
        self.assertEqual(b, newBlock)



        pass

    def test_getByID_storage_obj(self):
        pass