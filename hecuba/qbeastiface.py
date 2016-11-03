from qbeastIntegration import QbeastMaster
from qbeastIntegration.ttypes import entryPoint
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from multiprocessing import Process
from cassandra.metadata import Murmur3Token
import random

import sys

class QbeastIface():
    def __init__(self):
        from cassandra.cluster import Cluster
        self.c=Cluster(contact_points=['172.20.10.1'], port=19042)
        self.session=self.c.connect()
        self.meta=self.c.metadata

    def initQuery(self, selects, keyspace, table, area, precision, maxResults, tokens):
        print "QbeastIface initQuery"
        """
        Parameters:
        - selects
        - keyspace
        - table
        - area
        - precision
        - maxResults
        - tokens
        """
        entries = dict(map(lambda a: (
			a,
			entryPoint( self.meta.get_replicas("wordcount13",str(a)).pop , int(random.randint(26000,26999)) )
                       ),
                        tokens))
        # print "entries:", entries
        # return QueryLocation("ciao",entries)
        return 'QueryLocationResult'


def createServer():
    handler = QbeastIface()
    processor = QbeastMaster.Processor(handler)
    transport = TSocket.TServerSocket(port=2600)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
	 
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
	 
    print "Starting master server..."
    server.serve()

if __name__ == "__main__":
    p = Process(target=createServer, args=())
    p.start()



'''
queryLocation=server.initQuery()
queryLocation.tokens={"minerva-1":26902}
for key,value in tokens:
    os.system("ssh "+key+" ./qbeast-worker.py & disown "+value)
'''
