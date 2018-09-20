import unittest
from uuid import uuid4, UUID

import struct
from mock import Mock
from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket
from thrift.transport import TTransport

from hecuba.qthrift import QbeastMaster
from hecuba.qthrift import QbeastWorker
from hecuba.qthrift.ttypes import FilteringArea, Result, ColumnMeta, BasicTypes
from tests.withQbeast import TestServers


class BareThriftTests(unittest.TestCase):
    def setUp(self):
        self.ss = TestServers(100)

    def tearDown(self):
        self.ss.stopServers()

    def simple_init_test(self):

        transport = TSocket.TSocket("localhost", 2600)
        # Buffering is critical. Raw sockets are very slow
        transport = TTransport.TFramedTransport(transport)
        # Wrap in a protocol
        protocol = TBinaryProtocol.TBinaryProtocol(transport)
        transport.open()
        # Create a client to use the protocol encoder
        client = QbeastMaster.Client(protocol)
        area = FilteringArea([-.5, -.5, -.5], [3, 3, 3])
        partition = uuid4()

        qquid = client.initQuery(['time', 'partid', 'x', 'y', 'z'], 'test_qbeast',
                                 'particle_particle_idx_d8tree', area, 0.1, 100,
                                 [str(partition)])
        try:
            UUID(qquid)
        except:
            self.fail("Wrong qquid")

    def simple_get_test(self):

        workerT = TTransport.TFramedTransport(TSocket.TSocket("localhost", 2688))

        # Buffering is critical. Raw sockets are very slow

        # Wrap in a protocol
        pw = TBinaryProtocol.TBinaryProtocol(workerT)

        workerT.open()
        # Create a client to use the protocol encoder
        worker = QbeastWorker.Client(pw)

        read_more = True
        read = 0
        while read_more:
            result = worker.get("anything", 100, 200)
            if not result.hasMore:
                read_more = False
            read += len(result.data)
            self.assertTrue(len(result.data) == 100)
        self.assertEqual(read, 100)

