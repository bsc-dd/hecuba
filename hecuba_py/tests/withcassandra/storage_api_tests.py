import unittest
import uuid
import logging

from storage.api import getByID, TaskContext, start_task, end_task
from hecuba import config, StorageDict

from hecuba.partitioner import _dynamic_part_table_cql

from ..app.words import Words


class ApiTestSDict(StorageDict):
    '''
    @TypeSpec dict<<key:int>, value:double>
    '''


select_time = "SELECT * FROM hecuba.partitioning"


class PyCOMPSsArg:
    def __init__(self, storage_id):
        self.key = str(storage_id)


class StorageApi_Tests(unittest.TestCase):
    def class_type_test(self):
        base_dict = ApiTestSDict('test.api_sdict')
        # PyCOMPSs requires uuid of type str
        storage_id = str(base_dict.storage_id)
        del base_dict

        rebuild_dict = getByID(storage_id)
        self.assertTrue(isinstance(rebuild_dict, ApiTestSDict))

    def object_id_is_str_test(self):
        memory_obj = ApiTestSDict()

        self.assertTrue(hasattr(memory_obj, 'getID'))
        self.assertIsInstance(memory_obj.getID(), str, "PyCOMPSs specs states that getID should return a string")

        pers_dict = ApiTestSDict('test.api_id_str')
        self.assertTrue(hasattr(pers_dict, 'getID'))
        self.assertIsInstance(pers_dict.getID(), str, "PyCOMPSs specs states that getID should return a string")

    def test_getByID_block(self):
        # ki = KeyIter('testspace', 'tt', 'app.words.Words', 'fake-id', ['position'])
        from hecuba import config
        config.session.execute("DROP TABLE IF EXISTS my_app.so")
        config.session.execute("DROP TABLE IF EXISTS my_app.so_0")
        config.session.execute("DROP TABLE IF EXISTS my_app.so_1")
        SO = Words('so')
        b = next(SO.split())
        new_block = getByID(b.storage_id)
        self.assertEqual(b.storage_id, new_block.storage_id)
        self.assertEqual(b, new_block)

    def test_getByID_storage_obj(self):
        b = Words('testspace.tt')
        new_block = getByID(b.storage_id)
        self.assertEqual(b, new_block)

    def test_get_by_id_uuid(self):
        base_dict = ApiTestSDict('test.api_sdict')
        storage_id = base_dict.storage_id
        del base_dict

        rebuild_dict = getByID(storage_id)
        self.assertTrue(isinstance(rebuild_dict, ApiTestSDict))

    def test_get_by_id_str(self):
        base_dict = ApiTestSDict('test.api_sdict')
        storage_id = str(base_dict.storage_id)
        del base_dict

        rebuild_dict = getByID(storage_id)
        self.assertTrue(isinstance(rebuild_dict, ApiTestSDict))

    def test_get_by_id_getID(self):
        base_dict = ApiTestSDict('test.api_sdict')
        storage_id = base_dict.getID()
        del base_dict

        rebuild_dict = getByID(storage_id)
        self.assertTrue(isinstance(rebuild_dict, ApiTestSDict))

    def test_start_task_uuid(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        config.session.execute(_dynamic_part_table_cql)

        storage_id = uuid.uuid4()

        start_task([PyCOMPSsArg(storage_id)])

        inserted = list(config.session.execute(select_time))
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0].storage_id, storage_id)
        self.assertNotEqual(inserted[0].start_time, None)

    def test_end_task_uuid(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        config.session.execute(_dynamic_part_table_cql)

        storage_id = uuid.uuid4()

        end_task([PyCOMPSsArg(storage_id)])

        inserted = list(config.session.execute(select_time))
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0].storage_id, storage_id)
        self.assertNotEqual(inserted[0].end_time, None)

    def test_task_context_uuid(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        config.session.execute(_dynamic_part_table_cql)

        storage_id = uuid.uuid4()

        task_context = TaskContext(logger=logging, values=[PyCOMPSsArg(storage_id)])
        task_context.__enter__()
        task_context.__exit__(type=None, value=None, traceback=None)

        inserted = list(config.session.execute(select_time))
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0].storage_id, storage_id)
        self.assertNotEqual(inserted[0].start_time, None)
        self.assertNotEqual(inserted[0].end_time, None)

    def test_start_task_key(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        config.session.execute(_dynamic_part_table_cql)
        storage_id = uuid.uuid4()

        start_task([PyCOMPSsArg(storage_id)])

        inserted = list(config.session.execute(select_time))
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0].storage_id, storage_id)
        self.assertNotEqual(inserted[0].start_time, None)

    def test_end_task_key(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        config.session.execute(_dynamic_part_table_cql)

        storage_id = uuid.uuid4()

        end_task([PyCOMPSsArg(storage_id)])

        inserted = list(config.session.execute(select_time))
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0].storage_id, storage_id)
        self.assertNotEqual(inserted[0].end_time, None)

    def test_task_context_key(self):
        config.session.execute("DROP TABLE IF EXISTS hecuba.partitioning")
        config.session.execute(_dynamic_part_table_cql)

        storage_id = uuid.uuid4()

        task_context = TaskContext(logger=logging, values=[PyCOMPSsArg(storage_id)])
        task_context.__enter__()
        task_context.__exit__(type=None, value=None, traceback=None)

        inserted = list(config.session.execute(select_time))
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0].storage_id, storage_id)
        self.assertNotEqual(inserted[0].start_time, None)
        self.assertNotEqual(inserted[0].end_time, None)


if __name__ == "__main__":
    unittest.main()
