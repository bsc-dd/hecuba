import unittest
import uuid
import time
import numpy as np
from hecuba import hfetch, config

ksp = "hfetch_tests"


class StorageObjTest(unittest.TestCase):

    def test_hcache_init(self):
        table = "hcache_mix"
        query = "CREATE KEYSPACE IF NOT EXISTS {} " \
                "WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1 }} " \
                "AND durable_writes = 'true';".format(ksp)
        config.session.execute(query)
        config.session.execute("CREATE TABLE IF NOT EXISTS {}.{} (first int, second int, third text, val int, "
                               "PRIMARY KEY ((first, second ),third)) ;".format(ksp, table))

        # Random uuid
        storage_id = uuid.uuid4()

        tokens = []

        key_names = ["first", "second", "third"]
        values_names = [("val", "int")]
        values_names = map(lambda x: {"name": x[0], "type": x[1]}, values_names)

        hcache_args = (ksp, table, storage_id, tokens, key_names, values_names,
                       {'cache_size': config.max_cache_size,
                        'writer_par': config.write_callbacks_number,
                        'writer_buffer': config.write_buffer_size})

        hcache = hfetch.Hcache(*hcache_args)


        def test_get_row():
            hcache.get_row([1, 2, "3"])

        self.assertRaises(KeyError, test_get_row)

        keys = [1,2,"3"]
        values = [4]

        hcache.put_row(keys, values)

        result = hcache.get_row(keys)

        self.assertEqual(result, values)

        hcache.delete_row(keys)
        # self.assertEqual('tt1', nopars._table)

        query = "TRUNCATE TABLE {}.{};".format(ksp, table)
        config.session.execute(query)


    def test_numpys_config(self):
        table = "hcache_np_tests"
        query = "CREATE KEYSPACE IF NOT EXISTS {} " \
                "WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1 }} " \
                "AND durable_writes = 'true';".format(ksp)

        config.session.execute(query)

        config.session.execute('CREATE TABLE IF NOT EXISTS ' + ksp + '.' + table +
                               '(storage_id uuid , '
                               'cluster_id int, '
                               'block_id int, '
                               'payload blob, '
                               'PRIMARY KEY((storage_id,cluster_id),block_id))')

        # Random uuid
        storage_id = uuid.uuid4()

        tokens = []

        key_names = ['storage_id', 'cluster_id', 'block_id']
        values_names = [{'name': "payload", 'type': 'numpy', 'write_buffer_size': 100, 'write_callbacks_number': 8}]

        hcache_args = (ksp, table, storage_id, tokens, key_names, values_names,
                       {'cache_size': config.max_cache_size,
                        'writer_par': config.write_callbacks_number,
                        'writer_buffer': config.write_buffer_size})

        hcache = hfetch.Hcache(*hcache_args)

        nelem = 1024
        array = np.arange(nelem * nelem).reshape(nelem, nelem)
        hcache.put_row([storage_id, -1, -1],[array])

        del hcache

        hcache = hfetch.Hcache(*hcache_args)
        result = hcache.get_row([storage_id, -1, -1])

        self.assertEqual(len(result), 1)
        returned_array = result[0]

        self.assertTrue(np.array_equal(array, returned_array))


if __name__ == '__main__':
    unittest.main()
