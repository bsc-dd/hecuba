import struct
import unittest
from uuid import uuid4

from mock import Mock

from hecuba.qbeast import IndexedIterValue
from hecuba.qthrift.ttypes import Result, ColumnMeta, BasicTypes
from tests.withQbeast import wrapWorker


class IterThriftTests(unittest.TestCase):
    def simple_get_test(self):

        partition = uuid4()
        from hecuba.qthrift.QbeastWorker import Iface as WorkerIface
        m_mock = WorkerIface()

        r = Result(hasMore=False, count=10,
                   metadata={0: ColumnMeta(columnName='key', type=BasicTypes.INT),
                             1: ColumnMeta(columnName='value', type=BasicTypes.TEXT)},
                   data=[{0: struct.pack('I', i), 1: struct.pack("i", len(str(i)))+struct.pack("s", str(i))} for i in range(10)])
        m_mock.get = Mock(return_value=r)
        w = wrapWorker(m_mock)

        it = IndexedIterValue(partition, "localhost", lambda a:a)

        read = 0
        results = set()
        for i in it:
            read += 1
            results.add(i[1])
        self.assertEquals(read, 10)
        self.assertEqual(set(map(str, range(10))), results)
        w[0].serverTransport.close()
