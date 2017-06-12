import unittest

from hecuba.storageobj import StorageObj
from hecuba import hecuba_filter, python_filter


class TestIndexObj(StorageObj):
    '''
    @ClassField particles dict<<partid:int,time:float>,x:float,y:float,z:float>
    @Index_on particles x,y,z
    '''
    pass


class StorageObjFilterTest(unittest.TestCase):
    def test_filter_so(self):
        so = TestIndexObj().particles
        so.make_persistent('test.particle')
        la = lambda ((pid, time), (x, y, z)): x > 0 and x < 3 and y > 0 and y < 3 and z > 0 and z < 3
        filtered = hecuba_filter(
            lambda ((pid, time), (x, y, z)): x > 0 and x < 3 and y > 0 and y < 3 and z > 0 and z < 3,
            so.iteritems())
        c1 = len(python_filter(la, so.iteritems()))
        c2 = len(filtered)
        print 'so=>', c1, 'filtered =>', c2

        self.assertEqual(c1, c2)

    def test_filter_sampled(self):
        so = TestIndexObj().particles
        so.make_persistent('test.particle')
        from random import random
        la = lambda ((pid, time), (x, y, z)): x > 0 and x < 3 and y > 0 and y < 3 and z > 0 and z < 3 and random() < 0.2
        filtered = hecuba_filter(
            lambda ((pid, time), (x, y, z)): x > 0 and x < 3 and y > 0 and y < 3 and z > 0 and z < 3 and random() < 0.2,
            so.iteritems())
        c1 = len(python_filter(la, so.iteritems()))
        c2 = len(filtered)
        print 'so=>', c1, 'filtered =>', c2

        self.assertEqual(filtered._qbeast_meta.precision, 0.2)
        self.assertEqual(filtered._qbeast_meta.from_point, [0, 0, 0])
        self.assertEqual(filtered._qbeast_meta.to_point, [3, 3, 3])
        self.assertEqual(c1, c2)

    def test_filter_sampled2(self):
        so = TestIndexObj().particles
        so.make_persistent('test.particle')
        import random
        la = lambda \
            ((pid, time), (x, y, z)): x > 0 and x < 3 and y > 0 and y < 3 and z > 0 and z < 3 and random.random() < 0.2
        filtered = hecuba_filter(
            lambda
                ((pid, time), (x, y, z)): x > 0 and x < 3 and y > 0 and y < 3 and z > 0 and z < 3 and random.random() < 0.2,
            so.iteritems())
        c1 = len(python_filter(la, so.iteritems()))
        c2 = len(filtered)
        print 'so=>', c1, 'filtered =>', c2

        self.assertEqual(filtered._qbeast_meta.precision, 0.2)
        self.assertEqual(filtered._qbeast_meta.from_point, [0, 0, 0])
        self.assertEqual(filtered._qbeast_meta.to_point, [3, 3, 3])

        self.assertEqual(c1, c2)

    def test_filter_with_mem_filter(self):
        '''

        _index_vars = re.compile('lambda +\(\(([A-z, ]+)\), *\(([A-z, ]+)\)\) *:([^,]*)')
        keys, values, filter_body = m.groups()
        key_parameters = keys.replace(" ", '').split(',')
        value_parameters = values.replace(" ", '').split(',')
        for key in far_values.keys():
         inspected_function = str(inspected_function).replace(key, far_values[key])

        :return:
        '''
        so = TestIndexObj().particles
        so.make_persistent('test.particle')
        la = lambda \
                ((pid, time), (x, y, z)): x > 0 and x < 3 and y > 0 and y < 3 and z > 0 and z < 3 and pid & 1
        filtered = hecuba_filter(
            lambda
                ((pid, time), (x, y, z)): x > 0 and x < 3 and y > 0 and y < 3 and z > 0 and z < 3 and pid & 1,
            so.iteritems())
        c1 = len(python_filter(la, so.iteritems()))
        c2 = len(filtered)
        print 'so=>', c1, 'filtered =>', c2

        self.assertEqual(filtered._qbeast_meta.mem_filter, "pid & 1")
        self.assertEqual(c1, c2)

    def test_normal_filter(self):
        self.assertEqual(10, len(filter(lambda x: x > 10, range(0, 21))))


if __name__ == '__main__':
    unittest.main()
