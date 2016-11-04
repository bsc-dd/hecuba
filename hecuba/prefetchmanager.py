# author: G. Alomar
from cassandra.concurrent import execute_concurrent_with_args

from hecuba.settings import session,cluster
from cassandra.cluster import Cluster
from multiprocessing import Lock
from multiprocessing import Process
from multiprocessing import Pipe
from hecuba.settings import *
import random
import time


def pipeloop(pipeq, piper):
    global session, cluster
    try:
        session.shutdown()
    except Exception as e:
        print "error 1 in prefetch:", e
    try:
        cluster.shutdown()
    except Exception as e:
        print "error 2 in prefetch:", e
    try:
        cluster = Cluster(contact_points=contact_names, port=nodePort, protocol_version=2)
    except Exception as e:
        print "error 3 in prefetch:", e
    try:
        session = cluster.connect()
    except Exception as e:
        print "error 4 in prefetch:", e

    end = False

    while True:

        input_data = pipeq.recv()

        if input_data[0] == 'connect':
            '''
            input_data[0]=command
            input_data[1]=ips
            input_data[2]=port
            input_data[3]=query
            input_data[4]=ranges
            input_data[5]=colnames (cassandra table key and value)
            '''
            query = input_data[3]
            try:
                cluster = Cluster(contact_points=input_data[1], port=input_data[2], protocol_version=2)
            except Exception as e:
                print "Error creating cluster in pipeloop:", e
            colnames = input_data[5]
            done = False
            sleeptime = 0.5
            while not done:
                sessionexecute = 1
                while sessionexecute < 12:
                    try:
                        cluster.connect_timeout = 30
                        cluster.control_connection_timeout = 30
                        session = cluster.connect()
                        sessionexecute = 12
                        done = True
                    except Exception as e:
                        time.sleep(sleeptime)
                        if sleeptime < 20:
                            sleeptime = sleeptime * 1.5 * random.uniform(1.0, 3.0)
                        else:
                            sleeptime /= 2
                        if sessionexecute == 11:
                            raise e
                    sessionexecute += 1
                    statement = session.prepare(query)
                    statement.fetch_size = 100
                    # calculate ranges
                    token_ranges = []
                    metadata = cluster.metadata
                    ringtokens = metadata.token_map
                    ranges = str(input_data[4]).split('_')
                    for range in ranges:
                        for i, token in enumerate(ringtokens.ring):
                            if long(i) == long(range):
                                if not (long(i) == len(ringtokens.ring) - 1):
                                    start_token = token.value
                                    end_token = (ringtokens.ring[i+1]).value
                                    token_ranges.append((long(start_token), long(end_token)))
                                else:
                                    start_token = -9223372036854775808
                                    end_token = (ringtokens.ring[0]).value
                                    token_ranges.append((long(start_token), long(end_token)))
                                    start_token = (ringtokens.ring[i]).value
                                    end_token = 9223372036854775807
                                    token_ranges.append((long(start_token), long(end_token)))
            '''
            prepare the query and execute concurrent. Use the result in the loop implemented under the query command
            '''

        elif input_data[0] == 'query':
            r = {}
            r[0] = execute_concurrent_with_args(session, statement, token_ranges, 1)

        elif input_data[0] == 'continue':
            '''
            input_data[0]='query'
            implements the loop to access the result of the query executed in the connect command.
            stops iterating when reaching the fetch size limits and tries to receive the following command
            if the next command is a new 'query' continues iterating until reaching again the limit. If it is a terminate command closes the connection
            '''
            values = []
            if end:
                piper.send(values)
            else:
                value1 = ''
                value2 = ''
                for i, val in enumerate(r):
                    ind = 0
                    for (success, result) in r[i]:
                        if success:
                            for entry in result:
                                ind += 1
                                exec("value1 = entry." + colnames[0])
                                exec("value2 = entry." + colnames[1])
                                values.append((value1, value2))
                                if ind == statement.fetch_size:
                                    piper.send(values)
                                    values = []
                                    input_data = pipeq.recv()
                                    ind = 0
                end = True
                piper.send(values)

        elif input_data[0] == 'terminate':
            session.shutdown()
            cluster.shutdown()
            import sys
            sys.exit(1)


class PrefetchManager(object):

    chunksize = 0
    concurrency = 0
    p = {}
    pipeq_write = {}
    piper_write = {}
    pipeq_read = {}
    piper_read = {}
    query0 = ''
    query1 = ''
    query2 = ''

    def __init__(self, chunksize, concurrency, block):
        self.chunksize = chunksize
        self.concurrency = concurrency

        self.query = "SELECT * FROM " + block.keyspace + ".\"" + block.table_name + "\" WHERE token(" + block.key_names + ") >= ? AND token(" + block.key_names + ") < ?"
        for i in range(0, self.concurrency):
            self.pipeq_write[i], self.pipeq_read[i] = Pipe()
            self.piper_write[i], self.piper_read[i] = Pipe()
            self.p[i] = Process(target=pipeloop, args=(self.pipeq_read[i], self.piper_write[i]))
            self.p[i].start()
        for i in range(0, self.concurrency):
            lock = Lock()
            lock.acquire()

            persistentdict = block.storageobj
            keys = persistentdict.keyList[persistentdict.__class__.__name__]
            exec("keynames = persistentdict." + str(keys[0]) + ".dict_keynames")
            exec("dictname = persistentdict." + str(keys[0]) + ".dict_name")
            self.pipeq_write[i].send(['connect', contact_names, nodePort, self.query, block.token_ranges, [keynames, dictname]])
            lock.release()

    def read(self):
        toAppend = []
        for i in range(0, self.concurrency):
            toAppend.append(self.piper_read[i].recv())
        return toAppend

    def terminate(self):
        for i in range(0, self.concurrency):
            try:
                self.pipeq_write[i].send(['terminate'])
            except:
                print "not terminating correctly"

    def joinP(self):
        for k,v in self.p.iteritems():
            v.join()
