import unittest
import uuid
import logging

from storage.api import getByID, TaskContext, start_task, end_task
from hecuba import config, StorageDict


class ApiTestSDict(StorageDict):
    '''
    @TypeSpec dict<<key:int>, value:double>
    '''


select_time = "SELECT * FROM hecuba.partitioning"
create_partitioning = """CREATE TABLE IF NOT EXISTS hecuba.partitioning(
                         partitioning_uuid uuid,
                         storage_id uuid,
                         number_of_partitions int,
                         start_time double,
                         end_time double,
                         PRIMARY KEY (storage_id))
                         WITH default_time_to_live = 86400"""


class SimpleObject:
    pass


class StorageApiTest(unittest.TestCase):

    def test_class_type(self):
        base_dict = ApiTestSDict('test.api_sdict')
        storage_id = base_dict.getID()
        del base_dict

        rebuild_dict = getByID(storage_id)
        self.assertTrue(isinstance(rebuild_dict, ApiTestSDict))

    def test_start_task_uuid(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        config.session.execute(create_partitioning)

        storage_id = uuid.uuid4()
        simple_obj = SimpleObject()
        simple_obj._storage_id = storage_id

        start_task([simple_obj])

        inserted = list(config.session.execute(select_time))
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0].storage_id, storage_id)
        self.assertNotEqual(inserted[0].start_time, None)

    def test_end_task_uuid(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        config.session.execute(create_partitioning)

        storage_id = uuid.uuid4()
        simple_obj = SimpleObject()
        simple_obj._storage_id = storage_id

        end_task([simple_obj])

        inserted = list(config.session.execute(select_time))
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0].storage_id, storage_id)
        self.assertNotEqual(inserted[0].end_time, None)

    def test_task_context_uuid(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        config.session.execute(create_partitioning)

        storage_id = uuid.uuid4()
        simple_obj = SimpleObject()
        simple_obj._storage_id = storage_id

        task_context = TaskContext(logger=logging, values=[simple_obj])
        task_context.__enter__()
        task_context.__exit__(type=None, value=None, traceback=None)

        inserted = list(config.session.execute(select_time))
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0].storage_id, storage_id)
        self.assertNotEqual(inserted[0].start_time, None)
        self.assertNotEqual(inserted[0].end_time, None)

    def test_start_task_key(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        config.session.execute(create_partitioning)

        storage_id = uuid.uuid4()
        simple_obj = SimpleObject()
        simple_obj.__dict__["key"] = str(storage_id)

        start_task([simple_obj])

        inserted = list(config.session.execute(select_time))
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0].storage_id, storage_id)
        self.assertNotEqual(inserted[0].start_time, None)

    def test_end_task_key(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        config.session.execute(create_partitioning)

        storage_id = uuid.uuid4()
        simple_obj = SimpleObject()
        simple_obj.__dict__["key"] = str(storage_id)

        end_task([simple_obj])

        inserted = list(config.session.execute(select_time))
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0].storage_id, storage_id)
        self.assertNotEqual(inserted[0].end_time, None)

    def test_task_context_key(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        config.session.execute(create_partitioning)

        storage_id = uuid.uuid4()
        simple_obj = SimpleObject()
        simple_obj.__dict__["key"] = str(storage_id)

        task_context = TaskContext(logger=logging, values=[simple_obj])
        task_context.__enter__()
        task_context.__exit__(type=None, value=None, traceback=None)

        inserted = list(config.session.execute(select_time))
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0].storage_id, storage_id)
        self.assertNotEqual(inserted[0].start_time, None)
        self.assertNotEqual(inserted[0].end_time, None)


if __name__ == "__main__":
    unittest.main()
