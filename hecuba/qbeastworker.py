from qbeastIntegration import QbeastWorker
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from multiprocessing import Process
from qbeastIntegration.ttypes import *
import struct
import random
import sys

class QbeastWorkerImpl():

    def __init__(self):
        print "In __init__ ##################################"
        self.count=0

    def get(self, queryUUID, endToken, returnedList, readMax):
        print "In get #######################################"
        if(self.count>3):
            hasmore=False
            self.count=0
        else:
            hasmore=True
            self.count+=1
        metadata={1:DataType('partid',BasicTypes.INT),
		  2:DataType('time',BasicTypes.DOUBLE),
		  3:DataType('x',BasicTypes.DOUBLE),
		  4:DataType('y',BasicTypes.DOUBLE),
		  5:DataType('z',BasicTypes.DOUBLE)}
        data=[{1:struct.pack('<i',random.randint(0,1000)),
	       2:struct.pack('<d',random.random()),
	       3:struct.pack('<d',random.random()),
	       4:struct.pack('<d',random.random()),
	       5:struct.pack('<d',random.random())} for i in range(10)]
        return Result(hasmore, 10,metadata,data)


def createServer(serverport):
    handler = QbeastWorkerImpl()
    processor = QbeastWorker.Processor(handler)
    transport = TSocket.TServerSocket(port=int(serverport))
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
	 
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
	 
    print "Starting worker server..."
    server.serve()

if __name__ == "__main__":
    p = Process(target=createServer, args=(int(sys.argv[1]),))
    p.start()

