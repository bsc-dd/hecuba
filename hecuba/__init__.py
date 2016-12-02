# author: G. Alomar
import os
import logging
logging.basicConfig()

# Set default logging handler to avoid "No handler found" warnings.
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logging.getLogger('hecuba').addHandler(NullHandler())

class Config:
    def __init__(self):
        self._configured = False

    def __getattr__(self, item):
        if not self._configured:
            self.reset()
            return self.__getaatribute__(self, item)
        else:
            raise AttributeError('attribute %d not found', item)

    def reset(self, mock_cassandra=False):
        if mock_cassandra:
            logging.info('configuring mock environment')
        else:
            logging.info('configuring production environment')

        try:
            self.nodePort = int(os.environ['NODE_PORT'])
            logging.info('NODE_PORT: %s', self.nodePort)
        except KeyError:
            logging.warn('using default NODE_PORT 9042')
            self.nodePort = 9042

        try:
            self.contact_names = os.environ['CONTACT_NAMES'].split(",")
            logging.info('CONTACT_NAMES: %s', str.join(" ", self.contact_names))
        except KeyError:
            logging.warn('using default contact point localhost')
            self.contact_names = ['localhost']

        if hasattr(config, '') in globals():
            logging.warn('Shutting down pre-existent sessions and cluster')
            try:
                self.session.shutdown()
                self.cluster.shutdown()
            except:
                logging.warn('error shutting down')
        if mock_cassandra:
            class clusterMock:
                pass

            class sessionMock:
                def execute(self, *args, **kwargs):
                    logging.info('called mock.session')
                    return []

            self.cluster = clusterMock()
            self.session = sessionMock()

        else:
            from cassandra.cluster import Cluster
            logging.info('Initializing global session')
            self.cluster = Cluster(contact_points=self.contact_names, port=self.nodePort)
            self.session = self.cluster.connect()
            self.session.execute(
                "CREATE KEYSPACE IF NOT EXISTS hecuba WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3 }")
            self.session.execute(
                'CREATE TABLE IF NOT EXISTS hecuba.blocks (blockid text, block_classname text,storageobj_classname text, tkns list<bigint>, ' +
                'entry_point text , port int, ksp text , tab text , dict_name text , obj_type text, PRIMARY KEY(blockid))')

        try:
            self.execution_name = os.environ['EXECUTION_NAME']
            logging.info('EXECUTION_NAME: %s', self.execution_name)
        except KeyError:
            self.execution_name = 'hecuba_app'
            logging.warn('using default EXECUTION_NAME: %s', self.execution_name)

        try:
            self.workers_per_node = int(os.environ['WORKERS_PER_NODE'])
            logging.info('WORKERS_PER_NODE: %s', self.workers_per_node)
        except KeyError:
            self.workers_per_node = 8
            logging.warn('using default WORKERS_PER_NODE: %s', self.workers_per_node)

        try:
            self.number_of_blocks = int(os.environ['NUMBER_OF_BLOCKS'])
            logging.info('NUMBER_OF_BLOCKS: %s', self.number_of_blocks)
        except KeyError:
            self.number_of_blocks = 1024
            logging.warn('using default NUMBER_OF_BLOCKS: %s', self.number_of_blocks)

        try:
            self.cache_activated = os.environ['CACHE_ACTIVATED'].lower() == 'true'
            logging.info('CACHE_ACTIVATED: %s', self.cache_activated)
        except KeyError:
            self.cache_activated = True
            logging.warn('using default RANGES_PER_BLOCK: %s', self.cache_activated)

        try:
            self.batch_size = int(os.environ['BATCH_SIZE'])
            logging.info('BATCH_SIZE: %s', self.batch_size)
        except KeyError:
            self.batch_size = 100
            logging.warn('using default BATCH_SIZE: %s', self.batch_size)

        try:
            self.max_cache_size = int(os.environ['MAX_CACHE_SIZE'])
            logging.info('MAX_CACHE_SIZE: %s', self.max_cache_size)
        except KeyError:
            self.max_cache_size = 100
            logging.warn('using default MAX_CACHE_SIZE: %s', self.max_cache_size)

        try:
            self.repl_factor = int(os.environ['REPLICA_FACTOR'])
            logging.info('REPLICA_FACTOR: %s', self.repl_factor)
        except KeyError:
            self.repl_factor = 1
            logging.warn('using default REPLICA_FACTOR: %s', self.repl_factor)

        try:
            self.repl_class = os.environ['REPLICATION_STRATEGY']
            logging.info('REPLICATION_STRATEGY: %s', self.repl_class)
        except KeyError:
            self.repl_class = "SimpleStrategy"
            logging.warn('using default REPLICATION_STRATEGY: %s', self.repl_class)

        try:
            self.statistics_activated = os.environ['STATISTICS_ACTIVATED'].lower() == 'true'
            logging.info('STATISTICS_ACTIVATED: %s', self.statistics_activated)
        except KeyError:
            self.statistics_activated = True
            logging.warn('using default STATISTICS_ACTIVATED: %s', self.statistics_activated)

        try:
            query = "CREATE KEYSPACE IF NOT EXISTS %s WITH REPLICATION = { 'class' : \'%s\', 'replication_factor' : %d};" \
                    % (self.execution_name, self.repl_class, self.repl_factor)
            self.session.execute(query)
        except Exception as e:
            print "Cannot create keyspace", e

        config._configured = True



global config
config = Config()



