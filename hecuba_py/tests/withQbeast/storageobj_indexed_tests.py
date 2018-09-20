import unittest

from hecuba.storageobj import StorageObj


class TestStorageIndexedArgsObj(StorageObj):
    '''
       @ClassField test dict<<position:int>,x:float,y:float,z:float>
       @Index_on test x,y,z
    '''
    pass


class StorageObjIndexedTest(unittest.TestCase):
    def test_parse_index_on(self):
        a = TestStorageIndexedArgsObj()
        self.assertEqual(a.test._indexed_args, ['x', 'y', 'z'])
        a.make_persistent('tparse.t1')
        from storage.api import getByID
        b = getByID(a.getID())
        self.assertEqual(b.test._indexed_args, ['x', 'y', 'z'])


if __name__ == '__main__':
    unittest.main()
