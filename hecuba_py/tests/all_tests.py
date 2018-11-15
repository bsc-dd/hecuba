import unittest


class MyTestCase(unittest.TestCase):
    loader = unittest.TestLoader()
    start_dir = '/home/bscuser/hecuba_src/hecuba_py/tests/storageobj_tests.py'
    suite = loader.discover(start_dir)
    runner = unittest.TextTestRunner()
    runner.run(suite)
