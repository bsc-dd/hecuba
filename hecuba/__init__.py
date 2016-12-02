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


class Config(object):
    def __init__(self):
        self._configured = False
        self._props = {}

    def __getattr__(self, item):
        if not self._configured:
            self.reset()
        return self._props[item]

    def reset(self, mock_cassandra=False):
        self._configured = True
        if mock_cassandra:
            logging.info('configuring mock environment')
        else:
            logging.info('configuring production environment')

        try:
            self._props['nodePort'] = int(os.environ['NODE_PORT'])
            logging.info('NODE_PORT: %s', self._props['nodePort'])
        except KeyError:
            logging.warn('using default NODE_PORT 9042')
            self._props['nodePort'] = 9042

        try:
            self._props['contact_names'] = os.environ['CONTACT_NAMES'].split(",")
            logging.info('CONTACT_NAMES: %s', str.join(" ", self._props['contact_names']))
        except KeyError:
            logging.warn('using default contact point localhost')
            self._props['contact_names'] = ['localhost']

        if hasattr(config, '') in globals():
            logging.warn('Shutting down pre-existent sessions and cluster')
            try:
                self._props['session'].shutdown()
                self._props['cluster'].shutdown()
            except:
                logging.warn('error shutting down')
        if mock_cassandra:
            class clusterMock:
                pass

            class sessionMock:
                def execute(self, *args, **kwargs):
                    logging.info('called mock.session')
                    return []

            self._props['cluster'] = clusterMock()
            self._props['session'] = sessionMock()

        else:
            from cassandra.cluster import Cluster
            logging.info('Initializing global session')
            self._props['cluster'] = Cluster(contact_points=self._props['contact_names'], port=self._props['nodePort'])
            self._props['session'] = self._props['cluster'].connect()
            self._props['session'].execute(
                "CREATE KEYSPACE IF NOT EXISTS hecuba WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3 }")
            self._props['session'].execute(
                'CREATE TABLE IF NOT EXISTS hecuba.blocks (blockid text, block_classname text,storageobj_classname text, tkns list<bigint>, ' +
                'entry_point text , port int, ksp text , tab text , dict_name text , obj_type text, PRIMARY KEY(blockid))')

        try:
            self._props['execution_name'] = os.environ['EXECUTION_NAME']
            logging.info('EXECUTION_NAME: %s', self._props['execution_name'])
        except KeyError:
            self._props['execution_name'] = 'hecuba_app'
            logging.warn('using default EXECUTION_NAME: %s', self._props['execution_name'])

        try:
            self._props['workers_per_node'] = int(os.environ['WORKERS_PER_NODE'])
            logging.info('WORKERS_PER_NODE: %s', self._props['workers_per_node'])
        except KeyError:
            self._props['workers_per_node'] = 8
            logging.warn('using default WORKERS_PER_NODE: %s', self._props['workers_per_node'])

        try:
            self._props['number_of_blocks'] = int(os.environ['NUMBER_OF_BLOCKS'])
            logging.info('NUMBER_OF_BLOCKS: %s', self._props['number_of_blocks'])
        except KeyError:
            self._props['number_of_blocks'] = 1024
            logging.warn('using default NUMBER_OF_BLOCKS: %s', self._props['number_of_blocks'])

        try:
            self._props['cache_activated'] = os.environ['CACHE_ACTIVATED'].lower() == 'true'
            logging.info('CACHE_ACTIVATED: %s', self._props['cache_activated'])
        except KeyError:
            self._props['cache_activated'] = True
            logging.warn('using default RANGES_PER_BLOCK: %s', self._props['cache_activated'])

        try:
            self._props['batch_size'] = int(os.environ['BATCH_SIZE'])
            logging.info('BATCH_SIZE: %s', self._props['batch_size'])
        except KeyError:
            self._props['batch_size'] = 100
            logging.warn('using default BATCH_SIZE: %s', self._props['batch_size'])

        try:
            self._props['max_cache_size'] = int(os.environ['MAX_CACHE_SIZE'])
            logging.info('MAX_CACHE_SIZE: %s', self._props['max_cache_size'])
        except KeyError:
            self._props['max_cache_size'] = 100
            logging.warn('using default MAX_CACHE_SIZE: %s', self._props['max_cache_size'])

        try:
            self._props['repl_factor'] = int(os.environ['REPLICA_FACTOR'])
            logging.info('REPLICA_FACTOR: %s', self._props['repl_factor'])
        except KeyError:
            self._props['repl_factor'] = 1
            logging.warn('using default REPLICA_FACTOR: %s', self._props['repl_factor'])

        try:
            self._props['repl_class'] = os.environ['REPLICATION_STRATEGY']
            logging.info('REPLICATION_STRATEGY: %s', self._props['repl_class'])
        except KeyError:
            self._props['repl_class'] = "SimpleStrategy"
            logging.warn('using default REPLICATION_STRATEGY: %s', self._props['repl_class'])

        try:
            self._props['statistics_activated'] = os.environ['STATISTICS_ACTIVATED'].lower() == 'true'
            logging.info('STATISTICS_ACTIVATED: %s', self._props['statistics_activated'])
        except KeyError:
            self._props['statistics_activated'] = True
            logging.warn('using default STATISTICS_ACTIVATED: %s', self._props['statistics_activated'])

        try:
            query = "CREATE KEYSPACE IF NOT EXISTS %s WITH REPLICATION = { 'class' : \'%s\', 'replication_factor' : %d};" \
                    % (self._props['execution_name'], self._props['repl_class'], self._props['repl_factor'])
            self._props['session'].execute(query)
        except Exception as e:
            print "Cannot create keyspace", e


global config
config = Config()
