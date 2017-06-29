import unittest
from uuid import UUID, uuid4

from cassandra.cluster import Cluster
from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket
from thrift.transport import TTransport

from hecuba.qthrift import QbeastMaster
from hecuba.qthrift import QbeastWorker
from hecuba.qthrift.ttypes import FilteringArea


class BareThriftTests(unittest.TestCase):
    def simple_get_test(self):
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
        cluster = Cluster()
        session = cluster.connect("hecuba")
        session.execute('INSERT INTO hecuba.istorage(storage_id,tokens) VALUES (%s,%s)',
                        (partition, [-(1 << 31), (1 << 31) - 1]))

        qquid = client.initQuery(['time', 'partid', 'x', 'y', 'z'], 'test_qbeast',
                                 'particle_particle_idx_d8tree', area, 0.1, 100,
                                 [str(partition)])

        try:
            UUID(qquid)
        except ValueError:
            self.fail("Wrong qquid format")

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
            result = worker.get(str(partition), 100, 200)
            if not result.hasMore:
                read_more = False
            read += len(result.data)
            self.assertTrue(len(result.data) <= 200)
        self.assertGreater(read, 0)
        print 'read ', read, 'elements'
