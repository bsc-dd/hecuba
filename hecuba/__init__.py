# author: G. Alomar
import os
import logging
from cassandra.cluster import Cluster
from cassandra.policies import RetryPolicy

# Set default log.handler to avoid "No handler found" warnings.

stderrLogger = logging.StreamHandler()
f = '%(filename)s: %(levelname)s: %(funcName)s(): %(lineno)d:\t%(message)s'
stderrLogger.setFormatter(logging.Formatter(f))

log = logging.getLogger('hecuba')
log.addHandler(stderrLogger)

if 'DEBUG' in os.environ and os.environ['DEBUG'].lower() == "true":
    log.setLevel(logging.DEBUG)
else:
    log.setLevel(logging.INFO)


class _NRetry(RetryPolicy):
    def __init__(self, time_to_retry=5):
        self.time_to_retry = time_to_retry

    def on_unavailable(self, query, consistency, required_replicas, alive_replicas, retry_num):
        if retry_num > self.time_to_retry:
            return self.RETHROW, None
        else:
            return self.RETHROW, None

    def on_write_timeout(self, query, consistency, write_type, required_responses, received_responses, retry_num):
        if retry_num > self.time_to_retry:
            return self.RETHROW, None
        else:
            return self.RETHROW, None

    def on_read_timeout(self, query, consistency, required_responses, received_responses, data_retrieved, retry_num):
        if retry_num > self.time_to_retry:
            return self.RETHROW, None
        else:
            return self.RETHROW, None


class Config:
    class __Config:
        def __init__(self):
            self.configured = False

    instance = __Config()

    def __getattr__(self, item):
        return getattr(Config.instance, item)

    def __init__(self, mock_cassandra=False):
        Config.reset(mock_cassandra=mock_cassandra)

    @staticmethod
    def reset(mock_cassandra=False):
        singleton = Config.instance
        if singleton.configured and singleton.mock_cassandra == mock_cassandra:
            log.info('setting down')
            return

        singleton.mock_cassandra = mock_cassandra
        log.info('setting up configuration with mock_cassandra = %s', mock_cassandra)

        singleton.configured = True

        if 'CREATE_SCHEMA' in os.environ:
            singleton.id_create_schema = int(os.environ['CREATE_SCHEMA'])
        else:
            singleton.id_create_schema = -1

        if mock_cassandra:
            log.info('configuring mock environment')
        else:
            log.info('configuring production environment')
        try:
            singleton.nodePort = int(os.environ['NODE_PORT'])
            log.info('NODE_PORT: %d', singleton.nodePort)
        except KeyError:
            log.warn('using default NODE_PORT 9042')
            singleton.nodePort = 9042

        try:
            singleton.contact_names = os.environ['CONTACT_NAMES'].split(",")
            log.info('CONTACT_NAMES: %s', str.join(" ", singleton.contact_names))
        except KeyError:
            log.warn('using default contact point localhost')
            singleton.contact_names = ['127.0.0.1']

        if hasattr(singleton, 'session'):
            log.warn('Shutting down pre-existent sessions and cluster')
            try:
                singleton.session.shutdown()
                singleton.cluster.shutdown()
            except:
                log.warn('error shutting down')
        try:
            singleton.repl_factor = int(os.environ['REPLICA_FACTOR'])
            log.info('REPLICA_FACTOR: %d', singleton.repl_factor)
        except KeyError:
            singleton.repl_factor = 1
            log.warn('using default REPLICA_FACTOR: %d', singleton.repl_factor)

        try:
            singleton.execution_name = os.environ['EXECUTION_NAME']
            log.info('EXECUTION_NAME: %s', singleton.execution_name)
        except KeyError:
            singleton.execution_name = 'hecuba'
            log.warn('using default EXECUTION_NAME: %s', singleton.execution_name)

        if mock_cassandra:
            class clusterMock:
                def __init__(self):
                    from cassandra.metadata import Metadata
                    self.metadata = Metadata()
                    self.metadata.rebuild_token_map("Murmur3Partitioner", {})

            class sessionMock:

                def execute(self, *args, **kwargs):
                    log.info('called mock.session')
                    return []

                def prepare(self, *args, **kwargs):
                    return self

                def bind(self, *args, **kwargs):
                    return self

            singleton.cluster = clusterMock()
            singleton.session = sessionMock()
        else:
            log.info('Initializing global session')
            try:
                singleton.cluster = Cluster(contact_points=singleton.contact_names, port=singleton.nodePort,
                                            default_retry_policy=_NRetry(5))
                singleton.session = singleton.cluster.connect()
                from hfetch import connectCassandra
                # connecting c++ bindings
                connectCassandra(singleton.contact_names, singleton.nodePort)
                if singleton.id_create_schema == -1:
                    singleton.session.execute(
                        "CREATE KEYSPACE IF NOT EXISTS " + singleton.execution_name + " WITH replication = {'class': 'SimpleStrategy',"
                                                                                      "'replication_factor': %d }" % singleton.repl_factor)
                    singleton.session.execute(
                        'CREATE TABLE IF NOT EXISTS ' + singleton.execution_name + '.istorage (storage_id uuid, '
                                                                                   'class_name text,name text, '
                                                                                   'istorage_props map<text,text>, '
                                                                                   'tokens list<frozen<tuple<bigint,bigint>>>, entry_point text, port int, '
                                                                                   'indexed_args list<text>, nonindexed_args list<text>, '
                                                                                   'primary_keys list<frozen<tuple<text,text>>>,'
                                                                                   'columns list<frozen<tuple<text,text>>>,'
                                                                                   'value_list list<text>, mem_filter text, '
                                                                                   'PRIMARY KEY(storage_id))')

            except Exception as e:
                log.error('Exception creating cluster session. Are you in a testing env? %s', e)

        try:
            singleton.workers_per_node = int(os.environ['WORKERS_PER_NODE'])
            log.info('WORKERS_PER_NODE: %d', singleton.workers_per_node)
        except KeyError:
            singleton.workers_per_node = 8
            log.warn('using default WORKERS_PER_NODE: %d', singleton.workers_per_node)

        try:
            singleton.number_of_blocks = int(os.environ['NUMBER_OF_BLOCKS'])
            log.info('NUMBER_OF_BLOCKS: %d', singleton.number_of_blocks)
        except KeyError:
            singleton.number_of_blocks = 32
            log.warn('using default NUMBER_OF_BLOCKS: %d', singleton.number_of_blocks)

        try:
            singleton.min_number_of_tokens = int(os.environ['MIN_NUMBER_OF_TOKENS'])
            log.info('MIN_NUMBER_OF_TOKENS: %d', singleton.min_number_of_tokens)
        except KeyError:
            singleton.min_number_of_tokens = 1024
            log.warn('using default MIN_NUMBER_OF_TOKENS: %d', singleton.min_number_of_tokens)

        try:
            singleton.batch_size = int(os.environ['BATCH_SIZE'])
            log.info('BATCH_SIZE: %d', singleton.batch_size)
        except KeyError:
            singleton.batch_size = 1
            log.warn('using default BATCH_SIZE: %d', singleton.batch_size)

        try:
            singleton.ranges_per_block = int(os.environ['RANGES_PER_BLOCK:'])
            log.info('RANGES_PER_BLOCK:: %d', singleton.ranges_per_block)
        except KeyError:
            singleton.ranges_per_block = 1
            log.warn('using default RANGES_PER_BLOCK: %d', singleton.ranges_per_block)

        try:
            singleton.max_cache_size = int(os.environ['MAX_CACHE_SIZE'])
            log.info('MAX_CACHE_SIZE: %d', singleton.max_cache_size)
        except KeyError:
            singleton.max_cache_size = 100
            log.warn('using default MAX_CACHE_SIZE: %d', singleton.max_cache_size)

        try:
            singleton.repl_class = os.environ['REPLICATION_STRATEGY']
            log.info('REPLICATION_STRATEGY: %s', singleton.repl_class)
        except KeyError:
            singleton.repl_class = "SimpleStrategy"
            log.warn('using default REPLICATION_STRATEGY: %s', singleton.repl_class)

        try:
            singleton.statistics_activated = os.environ['STATISTICS_ACTIVATED'].lower() == 'true'
            log.info('STATISTICS_ACTIVATED: %s', singleton.statistics_activated)
        except KeyError:
            singleton.statistics_activated = True
            log.warn('using default STATISTICS_ACTIVATED: %s', singleton.statistics_activated)

        try:
            singleton.prefetch_size = int(os.environ['PREFETCH_SIZE'])
            log.info('PREFETCH_SIZE: %s', singleton.prefetch_size)
        except KeyError:
            singleton.prefetch_size = 10000
            log.warn('using default PREFETCH_SIZE: %s', singleton.prefetch_size)

        try:
            singleton.write_buffer_size = int(os.environ['WRITE_BUFFER_SIZE'])
            log.info('WRITE_BUFFER_SIZE: %s', singleton.write_buffer_size)
        except KeyError:
            singleton.write_buffer_size = 1000
            log.warn('using default WRITE_BUFFER_SIZE: %s', singleton.write_buffer_size)

        try:
            singleton.write_callbacks_number = int(os.environ['WRITE_CALLBACKS_NUMBER'])
            log.info('WRITE_CALLBACKS_NUMBER: %s', singleton.write_callbacks_number)
        except KeyError:
            singleton.write_callbacks_number = 16
            log.warn('using default WRITE_CALLBACKS_NUMBER: %s', singleton.write_callbacks_number)

        if singleton.id_create_schema == -1:
            try:
                query = "CREATE KEYSPACE IF NOT EXISTS %s WITH REPLICATION = { 'class' : \'%s\'," \
                        "'replication_factor' : %d};" \
                        % (singleton.execution_name, singleton.repl_class, singleton.repl_factor)
                singleton.session.execute(query)
            except Exception as e:
                print "Cannot create keyspace", e

        singleton.create_cache = set()


global config
config = Config()
