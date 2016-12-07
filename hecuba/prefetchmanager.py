# author: G. Alomar
from multiprocessing import Lock
from multiprocessing import Pipe
from multiprocessing import Process

from cassandra.concurrent import execute_concurrent_with_args

from hecuba import config


def pipeloop(pipeq, piper):

    end = False

    while True:

        input_data = pipeq.recv()

        if input_data[0] == 'connect':
            print "prefetchManager connect"
            config.__init__()

            '''
            input_data[0]=command
            input_data[1]=ips
            input_data[2]=port
            input_data[3]=query
            input_data[4]=ranges
            input_data[5]=colnames (cassandra table key and value)
            '''
            query = input_data[3]
            print "prefetchManager connect query:   ", query
            ranges = input_data[4]
            print "prefetchManager connect ranges:  ", ranges
            colnames = input_data[5]
            print "prefetchManager connect colnames:", colnames
            done = False
            tries = 0
            while tries < 5:
                print "tries:", str(tries+1)
                try:
                    statement = config.session.prepare(query)
                    print "prefetchManager connect statement: ", statement
                    statement.fetch_size = 100
                    token_ranges = []
                    metadata = config.cluster.metadata
                    ringtokens = metadata.token_map
                    print "prefetchManager connect ringtokens:", ringtokens
                    ran = set(ranges)
                    last = ringtokens.ring[len(ringtokens.ring)-1]
                    for t in ringtokens.ring:
                        if t.value in ran:
                            token_ranges.append((last.value, t.value))
                        last = t
                except Exception as e:
                    print "prefetchManager connect error:", e
                tries += 1
            print "prefetchManager connect token_ranges:", token_ranges

            '''
            prepare the query and execute concurrent. Use the result in the loop implemented under the query command
            '''

        elif input_data[0] == 'query':
            print "prefetchManager query"
            r = {}
            r[0] = execute_concurrent_with_args(config.session, statement, token_ranges, 1)

        elif input_data[0] == 'continue':
            print "prefetchManager continue"
            '''
            input_data[0]='query'
            implements the loop to access the result of the query executed in the connect command.
            stops iterating when reaching the fetch size limits and tries to receive the following command
            if the next command is a new 'query' continues iterating until reaching again the limit.
            If it is a terminate command closes the connection
            '''
            values = []
            if end:
                piper.send(values)
            else:
                for i, val in enumerate(r):
                    ind = 0
                    for (success, result) in r[i]:
                        if success:
                            for entry in result:
                                ind += 1
                                value1 = getattr(entry, colnames[0])
                                value2 = getattr(entry, colnames[1])
                                values.append((value1, value2))
                                if ind == statement.fetch_size:
                                    piper.send(values)
                                    values = []
                                    input_data = pipeq.recv()
                                    ind = 0
                end = True
                piper.send(values)

        elif input_data[0] == 'terminate':
            config.session.shutdown()
            config.cluster.shutdown()
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
        print "prefetch Manager __init__"
        self.chunksize = chunksize
        self.concurrency = concurrency
        partition_key = block.storageobj._get_default_dict()._primary_keys[0][0]

        self.query = "SELECT * FROM " + block.keyspace + "." + block.table_name + " WHERE " \
                     "token(" + partition_key + ") >= ? AND token(" + partition_key + ") < ?"
        for i in range(0, self.concurrency):
            self.pipeq_write[i], self.pipeq_read[i] = Pipe()
            self.piper_write[i], self.piper_read[i] = Pipe()
            self.p[i] = Process(target=pipeloop, args=(self.pipeq_read[i], self.piper_write[i]))
            self.p[i].start()
        for i in range(0, self.concurrency):
            lock = Lock()
            lock.acquire()

            props = block.storageobj._persistent_props
            for dictname, dict_prop in props.iteritems():
                keynames = map(lambda a: a[0], dict_prop['primary_keys'])
                self.pipeq_write[i].send(['connect', config.contact_names, config.nodePort,
                                          self.query, block.token_ranges, [keynames, dictname]])
            lock.release()

    def read(self):
        to_append = []
        for i in range(0, self.concurrency):
            to_append.append(self.piper_read[i].recv())
        return to_append

    def terminate(self):
        for i in range(0, self.concurrency):
            try:
                self.pipeq_write[i].send(['terminate'])
            except Exception:
                print "not terminating correctly"
