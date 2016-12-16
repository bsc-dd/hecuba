# author: G. Alomar
import os
import logging
from cassandra.cluster import Cluster
from cassandra.policies import RetryPolicy


# Set default logging handler to avoid "No handler found" warnings.
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logging.getLogger('hecuba').addHandler(NullHandler())


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
            logging.info('setting down')
            return

        singleton.mock_cassandra = mock_cassandra
        logging.info('setting up configuration with mock_cassandra = %s', mock_cassandra)

        singleton.configured = True
        if mock_cassandra:
            logging.info('configuring mock environment')
        else:
            logging.info('configuring production environment')
        try:
            singleton.nodePort = int(os.environ['NODE_PORT'])
            logging.info('NODE_PORT: %d', singleton.nodePort)
        except KeyError:
            logging.warn('using default NODE_PORT 9042')
            singleton.nodePort = 9042

        try:
            singleton.contact_names = os.environ['CONTACT_NAMES'].split(",")
            logging.info('CONTACT_NAMES: %s', str.join(" ", singleton.contact_names))
        except KeyError:
            logging.warn('using default contact point localhost')
            singleton.contact_names = ['localhost']

        if hasattr(singleton, 'session'):
            logging.warn('Shutting down pre-existent sessions and cluster')
            try:
                singleton.session.shutdown()
                singleton.cluster.shutdown()
            except:
                logging.warn('error shutting down')
        try:
            singleton.repl_factor = int(os.environ['REPLICA_FACTOR'])
            logging.info('REPLICA_FACTOR: %d', singleton.repl_factor)
        except KeyError:
            singleton.repl_factor = 1
            logging.warn('using default REPLICA_FACTOR: %d', singleton.repl_factor)

        if mock_cassandra:
            class clusterMock:
                pass

            class sessionMock:

                def execute(self, *args, **kwargs):
                    logging.info('called mock.session')
                    return []

                def prepare(self, *args, **kwargs):
                    return self

                def bind(self, *args, **kwargs):
                    return self

            singleton.cluster = clusterMock()
            singleton.session = sessionMock()

        else:
            logging.info('Initializing global session')
            try:
                singleton.cluster = Cluster(contact_points=singleton.contact_names, port=singleton.nodePort,
                                            default_retry_policy=_NRetry(5))
                singleton.session = singleton.cluster.connect()
                singleton.session.execute(
                    "CREATE KEYSPACE IF NOT EXISTS hecuba WITH replication = {'class': 'SimpleStrategy',"
                    "'replication_factor': %d }" % singleton.repl_factor)
                singleton.session.execute(
                    'CREATE TABLE IF NOT EXISTS hecuba.blocks (blockid text, class_name text,storageobj_classname text,'
                    'tkns list<bigint>, entry_point text , port int, ksp text , tab text , object_id text ,'
                    'obj_type text, PRIMARY KEY(blockid))')
                singleton.session.execute(
                    'CREATE TABLE IF NOT EXISTS hecuba.storage_objs (object_id text, class_name text,  ' +
                    'ksp text , tab text , obj_type text, PRIMARY KEY(object_id))')
            except Exception as e:
                logging.error('Exception creating cluster session. Are you in a testing env?', e)

        try:
            singleton.execution_name = os.environ['EXECUTION_NAME']
            logging.info('EXECUTION_NAME: %s', singleton.execution_name)
        except KeyError:
            singleton.execution_name = 'hecuba_app'
            logging.warn('using default EXECUTION_NAME: %s', singleton.execution_name)

        try:
            singleton.workers_per_node = int(os.environ['WORKERS_PER_NODE'])
            logging.info('WORKERS_PER_NODE: %d', singleton.workers_per_node)
        except KeyError:
            singleton.workers_per_node = 8
            logging.warn('using default WORKERS_PER_NODE: %d', singleton.workers_per_node)

        try:
            singleton.number_of_blocks = int(os.environ['NUMBER_OF_BLOCKS'])
            logging.info('NUMBER_OF_BLOCKS: %d', singleton.number_of_blocks)
        except KeyError:
            singleton.number_of_blocks = 1024
            logging.warn('using default NUMBER_OF_BLOCKS: %d', singleton.number_of_blocks)

        try:
            singleton.cache_activated = os.environ['CACHE_ACTIVATED'].lower() == 'true'
            logging.info('CACHE_ACTIVATED: %s', singleton.cache_activated)
        except KeyError:
            singleton.cache_activated = True
            logging.warn('using default CACHE_ACTIVATED: %s', singleton.cache_activated)

        try:
            singleton.batch_size = int(os.environ['BATCH_SIZE'])
            logging.info('BATCH_SIZE: %d', singleton.batch_size)
        except KeyError:
            singleton.batch_size = 1
            logging.warn('using default BATCH_SIZE: %d', singleton.batch_size)

        try:
            singleton.max_cache_size = int(os.environ['MAX_CACHE_SIZE'])
            logging.info('MAX_CACHE_SIZE: %d', singleton.max_cache_size)
        except KeyError:
            singleton.max_cache_size = 100
            logging.warn('using default MAX_CACHE_SIZE: %d', singleton.max_cache_size)

        try:
            singleton.repl_class = os.environ['REPLICATION_STRATEGY']
            logging.info('REPLICATION_STRATEGY: %s', singleton.repl_class)
        except KeyError:
            singleton.repl_class = "SimpleStrategy"
            logging.warn('using default REPLICATION_STRATEGY: %s', singleton.repl_class)

        try:
            singleton.statistics_activated = os.environ['STATISTICS_ACTIVATED'].lower() == 'true'
            logging.info('STATISTICS_ACTIVATED: %s', singleton.statistics_activated)
        except KeyError:
            singleton.statistics_activated = True
            logging.warn('using default STATISTICS_ACTIVATED: %s', singleton.statistics_activated)

        try:
            singleton.prefetch_activated = os.environ['PREFETCH_ACTIVATED'].lower() == 'true'
        except KeyError:
            singleton.prefetch_activated = False
            logging.warn('using default PREFETCH_ACTIVATED: %s', singleton.prefetch_activated)

        try:
            query = "CREATE KEYSPACE IF NOT EXISTS %s WITH REPLICATION = { 'class' : \'%s\'," \
                    "'replication_factor' : %d};" \
                    % (singleton.execution_name, singleton.repl_class, singleton.repl_factor)
            singleton.session.execute(query)
        except Exception as e:
            print "Cannot create keyspace", e


global config
config = Config()
