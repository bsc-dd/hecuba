import logging
import os
import time

from cassandra.cluster import Cluster
from cassandra.policies import RetryPolicy, RoundRobinPolicy, TokenAwarePolicy

# Set default log.handler to avoid "No handler found" warnings.

stderrLogger = logging.StreamHandler()
f = '%(filename)s: %(levelname)s: %(funcName)s(): %(lineno)d:\t%(message)s'
stderrLogger.setFormatter(logging.Formatter(f))

log = logging.getLogger('hecuba')
log.addHandler(stderrLogger)

if 'DEBUG' in os.environ and os.environ['DEBUG'].lower() == "true":
    log.setLevel(logging.DEBUG)
elif 'HECUBA_LOG' in os.environ:
    log.setLevel(os.environ['HECUBA_LOG'].upper())
else:
    log.setLevel(logging.ERROR)


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
            ## intercepted : list of numpy intercepted calls
            self.intercepted = {}

    instance = __Config()

    def __getattr__(self, item):
        return getattr(Config.instance, item)

    def executequery_withretries(self, query):
        """
            Executes 'query' to cassandra. If the query fails (for example due
            to a timeout) it will resend the query a maximum of NRETRIES(5)
        """
        executed=False
        nretries=0
        while not executed:
            try:
                self.instance.session.execute(query)
                executed=True
            except Exception as ir:
                log.warn("Unable to execute %s %s/5", query, nretries)
                nretries+=1
                if nretries==5: #FIXME the number of retries should be configurable
                    log.error("Too many retries. Aborting. Unable to execute %s", query)
                    raise ir
                time.sleep(1)

    def executelocked(self,query):
        if self.instance.concurrent_creation:
            r=self.instance.session.execute(self.instance._query_to_lock,[query])
            if r[0][0]:
                self.executequery_withretries(query)
            else:
                # FIXME find a better way to do this instead of an sleep... describe?
                time.sleep(.300)
        else:
            self.executequery_withretries(query)

    def __init__(self):
        singleton = Config.instance
        if singleton.configured:
            log.info('setting down')
            return

        singleton.configured = True


        if 'HECUBA_ARROW' in os.environ:
            env_var = os.environ['HECUBA_ARROW'].lower()
            singleton.arrow_enabled = False if env_var == 'no' or env_var == 'false' else True
            log.info('HECUBA_ARROW: {}'.format(singleton.arrow_enabled))
        else:
            singleton.arrow_enabled = False
            log.warn('Arrow access is DISABLED [HECUBA_ARROW=%s]', singleton.arrow_enabled)

        if 'CONCURRENT_CREATION' in os.environ:
            if os.environ['CONCURRENT_CREATION']=='True':
                singleton.concurrent_creation = True
            else:
                singleton.concurrent_creation = False
            log.info('CONCURRENT_CREATION: %s', str(singleton.concurrent_creation))
        else:
            singleton.concurrent_creation = False
            log.warn('Concurrent creation is DISABLED [CONCURRENT_CREATION=False]')

        if 'LOAD_ON_DEMAND' in os.environ:
            if os.environ['LOAD_ON_DEMAND']=='False':
                singleton.load_on_demand = False
            else:
                singleton.load_on_demand = True
            log.info('LOAD_ON_DEMAND: %s', str(singleton.load_on_demand))
        else:
            singleton.load_on_demand = True
            log.warn('Load data on demand is ENABLED [LOAD_ON_DEMAND=True]')

        if 'CREATE_SCHEMA' in os.environ:
            env_var = os.environ['CREATE_SCHEMA'].lower()
            singleton.id_create_schema = False if env_var == 'no' or env_var == 'false' else True
            log.info('CREATE_SCHEMA: %d', singleton.id_create_schema)
        else:
            singleton.id_create_schema = True
            log.warn('Creating keyspaces and tables by default [CREATE_SCHEMA=True]')
        try:
            singleton.nodePort = int(os.environ['NODE_PORT'])
            log.info('NODE_PORT: %d', singleton.nodePort)
        except KeyError:
            log.warn('using default NODE_PORT 9042')
            singleton.nodePort = 9042

        try:
            singleton.contact_names = os.environ['CONTACT_NAMES'].split(",")
            log.info('CONTACT_NAMES: %s', str.join(" ", singleton.contact_names))
            # Convert node names to ips if needed
            import socket
            contact_names_ips = []
            show_translation = False
            for h_name in singleton.contact_names:
                IP_addres = socket.gethostbyname(h_name)
                if (IP_addres != h_name):
                    show_translation=True
                contact_names_ips.append(IP_addres)
            singleton.contact_names = contact_names_ips
            if show_translation:
                log.info('CONTACT_NAMES: %s', str.join(" ", singleton.contact_names))

        except KeyError:
            log.warn('using default contact point localhost')
            singleton.contact_names = ['127.0.0.1']

        try:
            singleton.kafka_names = os.environ['KAFKA_NAMES'].split(",")
            log.info('KAFKA_NAMES: %s', str.join(" ", singleton.kafka_names))
        except KeyError:
            log.warn('kakfa names defaults to %s', str.join(" ", singleton.contact_names))
            singleton.kafka_names = singleton.contact_names

        if hasattr(singleton, 'session'):
            log.warn('Shutting down pre-existent sessions and cluster')
            try:
                singleton.session.shutdown()
                singleton.cluster.shutdown()
            except Exception:
                log.warn('error shutting down')
        try:
            singleton.replication_factor = int(os.environ['REPLICA_FACTOR'])
            log.info('REPLICA_FACTOR: %d', singleton.replication_factor)
        except KeyError:
            singleton.replication_factor = 1
            log.warn('using default REPLICA_FACTOR: %d', singleton.replication_factor)

        try:
            user_defined_execution_name = os.environ['EXECUTION_NAME']
            if user_defined_execution_name == 'hecuba':
                raise RuntimeError('Error: the application keyspace cannot be \'hecuba\'. '
                                   'This keyspace is reserved for storing metadata.')
            singleton.execution_name = user_defined_execution_name
            log.info('EXECUTION_NAME: %s', singleton.execution_name)
        except KeyError:
            singleton.execution_name = 'my_app'
            log.warn('using default EXECUTION_NAME: %s', singleton.execution_name)
        try:
            singleton.splits_per_node = int(os.environ['SPLITS_PER_NODE'])
            log.info('SPLITS_PER_NODE: %d', singleton.splits_per_node)
        except KeyError:
            singleton.splits_per_node = 32
            log.warn('using default SPLITS_PER_NODE: %d', singleton.splits_per_node)

        try:
            singleton.token_range_size = int(os.environ['TOKEN_RANGE_SIZE'])
            log.info('TOKEN_RANGE_SIZE: %d', singleton.token_range_size)
            singleton.target_token_range_size = None
        except KeyError:
            singleton.token_range_size = None

            try:
                singleton.target_token_range_size = int(os.environ['TARGET_TOKEN_RANGE_SIZE'])
                log.info('TARGET_TOKEN_RANGE_SIZE: %d', singleton.target_token_range_size)
            except KeyError:
                singleton.target_token_range_size = 64 * 1024
                log.warn('using default TARGET_TOKEN_RANGE_SIZE: %d', singleton.target_token_range_size)

        try:
            singleton.max_cache_size = int(os.environ['MAX_CACHE_SIZE'])
            log.info('MAX_CACHE_SIZE: %d', singleton.max_cache_size)
        except KeyError:
            singleton.max_cache_size = 1000
            log.warn('using default MAX_CACHE_SIZE: %d', singleton.max_cache_size)

        try:
            singleton.replication_strategy = os.environ['REPLICATION_STRATEGY']
            log.info('REPLICATION_STRATEGY: %s', singleton.replication_strategy)
        except KeyError:
            singleton.replication_strategy = "SimpleStrategy"
            log.warn('using default REPLICATION_STRATEGY: %s', singleton.replication_strategy)

        try:
            singleton.replication_strategy_options = os.environ['REPLICATION_STRATEGY_OPTIONS']
            log.info('REPLICATION_STRATEGY_OPTIONS: %s', singleton.replication_strategy_options)
        except KeyError:
            singleton.replication_strategy_options = ""
            log.warn('using default REPLICATION_STRATEGY_OPTIONS: %s', singleton.replication_strategy_options)

        if singleton.replication_strategy == "SimpleStrategy":
            singleton.replication = "{'class' : 'SimpleStrategy', 'replication_factor': %d}" % \
                                    singleton.replication_factor
        else:
            singleton.replication = "{'class' : '%s', %s}" % (
                singleton.replication_strategy, singleton.replication_strategy_options)
        try:
            singleton.hecuba_print_limit = int(os.environ['HECUBA_PRINT_LIMIT'])
            log.info('HECUBA_PRINT_LIMIT: %s', singleton.hecuba_print_limit)
        except KeyError:
            singleton.hecuba_print_limit = 1000
            log.warn('using default HECUBA_PRINT_LIMIT: %s', singleton.hecuba_print_limit)

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

        try:
            env_var = os.environ['TIMESTAMPED_WRITES'].lower()
            singleton.timestamped_writes = False if env_var == 'no' or env_var == 'false' else True
            log.info('TIMESTAMPED WRITES ENABLED? {}'.format(singleton.timestamped_writes))
        except KeyError:
            singleton.timestamped_writes = False
            log.warn('using default TIMESTAMPED_WRITES: %s', singleton.timestamped_writes)

        if singleton.max_cache_size < singleton.write_buffer_size:
            import warnings
            message = "Defining a MAX_CACHE_SIZE smaller than WRITE_BUFFER_SIZE can result " \
                      "in reading outdated results from the persistent storage"
            warnings.warn(message)

        log.info('Initializing global session')

        singleton.cluster = Cluster(contact_points=singleton.contact_names,
                                    load_balancing_policy=TokenAwarePolicy(RoundRobinPolicy()),
                                    port=singleton.nodePort,
                                    default_retry_policy=_NRetry(5))
        singleton.session = singleton.cluster.connect()
        singleton.session.encoder.mapping[tuple] = singleton.session.encoder.cql_encode_tuple
        if singleton.concurrent_creation:
            configure_lock=[
                """CREATE KEYSPACE IF NOT EXISTS hecuba_locks
                        WITH replication=  {'class': 'SimpleStrategy', 'replication_factor': 1};
                """,
                """CREATE TABLE IF NOT EXISTS hecuba_locks.table_lock
                        (table_name text, PRIMARY KEY (table_name));
                """,
                "TRUNCATE table hecuba_locks.table_lock;"
            ]
            for query in configure_lock:
                try:
                    self.instance.session.execute(query)
                except Exception as e:
                    log.error("Error executing query %s" % query)
                    raise e
            singleton._query_to_lock=singleton.session.prepare("INSERT into hecuba_locks.table_lock (table_name) values (?) if not exists;")

        if singleton.id_create_schema:
            queries = [
                "CREATE KEYSPACE IF NOT EXISTS hecuba  WITH replication = %s" % singleton.replication,
                """CREATE TYPE IF NOT EXISTS hecuba.q_meta(
                mem_filter text, 
                from_point frozen<list<double>>,
                to_point frozen<list<double>>,
                precision float);
                """,
                """CREATE TYPE IF NOT EXISTS hecuba.np_meta (flags int, elem_size int, partition_type tinyint,
                dims list<int>, strides list<int>, typekind text, byteorder text)""",
                """CREATE TABLE IF NOT EXISTS hecuba
                .istorage (storage_id uuid, 
                class_name text,name text, 
                istorage_props map<text,text>, 
                tokens list<frozen<tuple<bigint,bigint>>>,
                indexed_on list<text>,
                qbeast_random text,
                qbeast_meta frozen<q_meta>,
                numpy_meta frozen<np_meta>,
                block_id int,
                base_numpy uuid,
                view_serialization blob,
                primary_keys list<frozen<tuple<text,text>>>,
                columns list<frozen<tuple<text,text>>>,
                PRIMARY KEY(storage_id));
                """,
                "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (singleton.execution_name, singleton.replication)]
            for query in queries:
                try:
                    self.executelocked(query)
                except Exception as e:
                    log.error("Error executing query %s" % query)
                    raise e

        from hecuba.hfetch import connectCassandra, HArrayMetadata
        # connecting c++ bindings
        connectCassandra(singleton.contact_names, singleton.nodePort)

        if singleton.id_create_schema:
            time.sleep(10)
        singleton.cluster.register_user_type('hecuba', 'np_meta', HArrayMetadata)
        # Create a dummy arrayDataStore to generate the TokensToHost variable


global config
config = Config()

from .parser import Parser
from .storageobj import StorageObj
from .hdict import StorageDict
from .hnumpy import StorageNumpy
from .storagestream import StorageStream
from .hfilter import hfilter

if not filter == hfilter:
    import builtins

    builtins.python_filter = filter
    builtins.filter = hfilter


def _intercept_numpy_method(method_name):
    """
    Intercept Numpy.'method_name' and use StorageNumpy.'method_name' instead.
    """
    if not isinstance(method_name, str):
        raise TypeError("Intercepted method name MUST be an string")

    # INTERCEPT Numpy METHODS
    import numpy as np

    if np.__dict__[method_name] != StorageNumpy.__dict__[method_name]:
        config.instance.intercepted[method_name] = np.__dict__[method_name]
        np.__dict__[method_name] = StorageNumpy.__dict__[method_name]
    else:
        #print("WARNING: method {} already intercepted".format(method_name), flush=True)
        if not method_name in config.instance.intercepted:
            print("WARNING: method {} already intercepted but we have lost the original method!!!".format(method_name), flush=True)
            # This only happens executing tests... we think that is due to the reload of hecuba, but not numpy (not 100% sure)
            # therefore our solution is to reload numpy.
            import importlib
            importlib.reload(np)
            config.instance.intercepted[method_name] = np.__dict__[method_name]
            np.__dict__[method_name] = StorageNumpy.__dict__[method_name]

_intercept_numpy_method('dot')
_intercept_numpy_method('array_equal')
_intercept_numpy_method('concatenate')

__all__ = ['StorageObj', 'StorageDict', 'StorageNumpy', 'StorageStream', 'Parser']
