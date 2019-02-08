import unittest
loader = unittest.TestLoader()
start_dir = '/home/bscuser/parser/hecuba/hecuba_py/tests/withcassandra'
suite = loader.discover(start_dir)

runner = unittest.TextTestRunner()
runner.run(suite)