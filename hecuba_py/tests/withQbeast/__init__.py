import uuid
from threading import Thread

import time
from multiprocessing import Process

import struct
from mock import Mock
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.transport import TSocket, TTransport

from hecuba.qthrift import QbeastMaster, QbeastWorker
from hecuba.qthrift.ttypes import Result, ColumnMeta, BasicTypes


class QbeastServerMock:
    def initQuery(self, selects, keyspace, table, area, precision, maxResults, blockIDs):
        return str(uuid.uuid4())


class QbeastWorkerMock:
    def get(self, blockID, returnAtLeast, readMax):
        return Result(hasMore=False, count=100, metadata={}, data=[])


class IgnoreServer:
    def __init__(self, server):
        self.server = server
        self.serverTransport = server.serverTransport

    def serve(self):
        try:
            self.server.serve()
        except:
            pass



def wrapMaster(impl):
    processor = QbeastMaster.Processor(impl)
    transport = TSocket.TServerSocket(port=2600)
    tfactory = TTransport.TFramedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = IgnoreServer(TServer.TThreadedServer(processor, transport, tfactory, pfactory))
    t = Process(target=server.serve)
    t.daemon = True
    t.start()
    return server, t


def wrapWorker(impl):
    processor = QbeastWorker.Processor(impl)
    transport = TSocket.TServerSocket(port=2688)
    tfactory = TTransport.TFramedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = IgnoreServer(TServer.TThreadedServer(processor, transport, tfactory, pfactory))
    t = Process(target=server.serve)
    t.daemon = True
    t.start()
    return server, t


def initQbeastMock(master_class=QbeastServerMock, worker_class=QbeastWorkerMock):
    handler = master_class()
    server_master = wrapMaster(handler)
    handler = worker_class()
    server_worker = wrapWorker(handler)
    return server_master, server_worker


class TestServers:
    def __init__(self, n_results=10, quiid=uuid.uuid4()):
        from hecuba.qthrift.QbeastMaster import Iface as MasterIface
        from hecuba.qthrift.QbeastWorker import Iface as WorkerIface
        m_mock = MasterIface()
        w_mock = WorkerIface()

        r = Result(hasMore=False, count=n_results,
                   metadata={0: ColumnMeta(columnName='partid', type=BasicTypes.INT),
                             1: ColumnMeta(columnName='time', type=BasicTypes.FLOAT),
                             2: ColumnMeta(columnName='x', type=BasicTypes.FLOAT),
                             3: ColumnMeta(columnName='y', type=BasicTypes.FLOAT),
                             4: ColumnMeta(columnName='z', type=BasicTypes.FLOAT)},
                   data=
                   [{0: struct.pack('!I', i),
                     1: struct.pack("!f", i * 0.1),
                     2: struct.pack("!f", i * 0.2),
                     3: struct.pack("!f", i * 0.3),
                     4: struct.pack("!f", i * 0.4)}
                    for i in range(n_results)]
                   )
        w_mock.get = Mock(return_value=r)
        self.w = wrapWorker(w_mock)

        m_mock.initQuery = Mock(return_value=str(quiid))
        self.m = wrapMaster(m_mock)

    def stopServers(self):
        self.w[1].terminate()
        self.m[1].terminate()
        self.w[0].serverTransport.close()
        self.m[0].serverTransport.close()
        time.sleep(10)  # freeing the ports
