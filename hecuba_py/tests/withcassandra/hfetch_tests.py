import unittest
import numpy as np

from hecuba import config, StorageDict
from hfetch import HArrayMetadata


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
        for value in range(last_value):
            my_dict[0] = value

        del my_dict
        import gc
        gc.collect()

        my_dict = ConcurrentDict("concurrent_dict")

        retrieved = my_dict[0]

        config.timestamped_writes = previous_cfg

        self.assertEqual(retrieved, last_value - 1)

    def test_harray_metadata_init(self):
        base = np.arange(7 * 8 * 9 * 10).reshape((7, 8, 9, 10))

        args = (list(base.shape), list(base.strides), [-1]*base.ndim, base.dtype.kind, base.dtype.byteorder,
                base.itemsize, base.flags.num, 0)

        obj = HArrayMetadata(*args)

        with self.assertRaises(TypeError):
            obj = HArrayMetadata()

        with self.assertRaises(TypeError):
            obj = HArrayMetadata(args[1:])

    def test_harray_metadata_refs(self):
        base = np.arange(10)
        args = (list(base.shape), list(base.strides), [-1], base.dtype.kind, base.dtype.byteorder,
                base.itemsize, base.flags.num, 0)

        obj = HArrayMetadata(*args)
        import gc
        gc.collect()
        import sys
        # The test has the first ref, the method getrefcount has the second reference
        self.assertEqual(sys.getrefcount(obj), 2)

    def test_register(self):
        from hfetch import HArrayMetadata
        # connecting c++ bindings
        from hecuba import config
        config.session.execute("DROP KEYSPACE IF EXISTS test_np_meta;")

        config.session.execute("CREATE KEYSPACE IF NOT EXISTS test_np_meta "
                               "WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")

        config.session.execute("""CREATE TYPE IF NOT EXISTS test_np_meta.np_meta (flags int, elem_size int, 
        partition_type tinyint, dims list<int>, strides list<int>, typekind text, byteorder text)""")

        config.cluster.register_user_type('test_np_meta', 'np_meta', HArrayMetadata)
        config.session.execute("DROP KEYSPACE IF EXISTS test_np_meta;")

