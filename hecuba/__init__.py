import os
import logging
from cassandra.cluster import Cluster
from cassandra.policies import RetryPolicy
import re

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
            except _:
                log.warn('error shutting down')
        try:
            singleton.repl_factor = int(os.environ['REPLICA_FACTOR'])
            log.info('REPLICA_FACTOR: %d', singleton.repl_factor)
        except KeyError:
            singleton.repl_factor = 1
            log.warn('using default REPLICA_FACTOR: %d', singleton.repl_factor)

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
                singleton.session.encoder.mapping[tuple] = singleton.session.encoder.cql_encode_tuple
                from hfetch import connectCassandra
                # connecting c++ bindings
                connectCassandra(singleton.contact_names, singleton.nodePort)
                if singleton.id_create_schema == -1:
                    singleton.session.execute(
                        ('CREATE KEYSPACE IF NOT EXISTS hecuba' +
                         " WITH replication = {'class': 'SimpleStrategy', "
                         "'replication_factor': %d }" % singleton.repl_factor))

                    singleton.session.execute('CREATE TYPE IF NOT EXISTS hecuba.q_meta('
                                              'mem_filter text, '
                                              'from_point frozen < list < float >>,'
                                              'to_point frozen < list < float >>,'
                                              'precision float)')

                    singleton.session.execute(
                        'CREATE TABLE IF NOT EXISTS hecuba' +
                        '.istorage (storage_id uuid, '
                        'class_name text,name text, '
                        'istorage_props map<text,text>, '
                        'tokens list<frozen<tuple<bigint,bigint>>>,'
                        'indexed_on list<text>,'
                        'entry_point text,'
                        'qbeast_id uuid,'
                        'qbeast_meta q_meta,'
                        'primary_keys list<frozen<tuple<text,text>>>,'
                        'columns list<frozen<tuple<text,text>>>,'
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
            singleton.number_of_partitions = int(os.environ['NUMBER_OF_BLOCKS'])
            log.info('NUMBER_OF_BLOCKS: %d', singleton.number_of_partitions)
        except KeyError:
            singleton.number_of_partitions = 32
            log.warn('using default NUMBER_OF_BLOCKS: %d', singleton.number_of_partitions)

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
            singleton.hecuba_print_limit = int(os.environ['HECUBA_PRINT_LIMIT'])
            log.info('HECUBA_PRINT_LIMIT: %s', singleton.hecuba_print_limit)
        except KeyError:
            singleton.hecuba_print_limit = 1000
            log.warn('using default HECUBA_PRINT_LIMIT: %s', singleton.hecuba_print_limit)

        try:
            singleton.hecuba_type_checking = os.environ['HECUBA_TYPE_CHECKING'].lower() == 'true'
            log.info('HECUBA_TYPE_CHECKING: %s', singleton.hecuba_type_checking)
        except KeyError:
            singleton.hecuba_type_checking = False
            log.warn('using default HECUBA_TYPE_CHECKING: %s', singleton.hecuba_type_checking)

        try:
            singleton.statistics_activated = os.environ['STATISTICS_ACTIVATED'].lower() == 'true'
            log.info('STATISTICS_ACTIVATED: %s', singleton.statistics_activated)
        except KeyError:
            singleton.statistics_activated = True
            log.warn('using default STATISTICS_ACTIVATED: %s', singleton.statistics_activated)

        try:
            singleton.hecuba_type_checking = os.environ['HECUBA_TYPE_CHECKING'].lower() == 'true'
            log.info('HECUBA_TYPE_CHECKING: %s', singleton.hecuba_type_checking)
        except KeyError:
            singleton.hecuba_type_checking = False
            log.warn('using default HECUBA_TYPE_CHECKING: %s', singleton.hecuba_type_checking)

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
            except Exception as ex:
                log.warn("Cannot create keyspace %s" % singleton.execution_name)
                raise ex

        try:
            singleton.qbeast_master_port = int(os.environ['QBEAST_MASTER_PORT'])
            log.info('QBEAST_MASTER_PORT: %d', singleton.qbeast_master_port)
        except KeyError:
            log.warn('using default qbeast master port 2600')
            singleton.qbeast_master_port = 2600

        try:
            singleton.qbeast_worker_port = int(os.environ['QBEAST_WORKER_PORT'])
            log.info('QBEAST_WORKER_PORT: %d', singleton.qbeast_worker_port)
        except KeyError:
            log.warn('using default qbeast worker port 2688')
            singleton.qbeast_worker_port = 2688

        try:
            singleton.qbeast_entry_node = os.environ['QBEAST_ENTRY_NODE'].split(",")
            log.info('QBEAST_ENTRY_NODE: %s', singleton.qbeast_entry_node)
        except KeyError:
            log.warn('using default qbeast entry node localhost')
            import socket
            singleton.qbeast_entry_node = [socket.gethostname()]

        try:
            singleton.qbeast_max_results = int(os.environ['QBEAST_MAX_RESULTS'].split(","))
            log.info('QBEAST_MAX_RESULTS: %d', singleton.qbeast_max_results)
        except KeyError:
            log.warn('using default qbeast max results 10000000')
            singleton.qbeast_max_results = 10000000

        try:
            singleton.qbeast_return_at_least = int(os.environ['RETURN_AT_LEAST'].split(","))
            log.info('QBEAST_RETURN_AT_LEAST: %d', singleton.qbeast_return_at_least)
        except KeyError:
            log.warn('using default qbeast return at least 100')
            singleton.qbeast_return_at_least = 100

        try:
            singleton.qbeast_read_max = int(os.environ['READ_MAX'].split(","))
            log.info('QBEAST_READ_MAX: %d', singleton.qbeast_read_max)
        except KeyError:
            log.warn('using default qbeast read max 10000')
            singleton.qbeast_read_max = 10000


filter_reg = re.compile(' *lambda *\( *\( *([\w, ]+) *\) *, *\( *([\w, ]+) *\) *\) *: *([\w<>().&*+/ ]+) *,')
random_reg = re.compile('(.*)((random.random\(\)|random\(\)) *< *([0.1]))(.*)')


def hecuba_filter(lambda_filter, iterable):
    if hasattr(iterable, '_storage_father') and hasattr(iterable._storage_father, '_indexed_args') \
            and iterable._storage_father._indexed_args is not None:
        indexed_args = iterable._storage_father._indexed_args
        father = iterable._storage_father
        import inspect
        from byteplay import Code
        func = Code.from_code(lambda_filter.func_code)
        if lambda_filter.func_closure is not None:
            vars_to_vals = zip(func.freevars, map(lambda x: x.cell_contents, lambda_filter.func_closure))
        inspected = inspect.findsource(lambda_filter)
        inspected_function = '\n'.join(inspected[0][inspected[1]:]).replace('\n', '')
        m = filter_reg.match(inspected_function)
        if m is not None:
            key_parameters, value_parameters, function_arguments = m.groups()
            key_parameters = re.sub(r'\s+', '', key_parameters).split(',')
            value_parameters = re.sub(r'\s+', '', value_parameters).split(',')
        initial_index_arguments = []
        m = random_reg.match(function_arguments)
        precision = 1.0
        precision_ind = -1
        if m is not None:
            params = m.groups()
            for ind, param in enumerate(params):
                if param == 'random()' or param == 'random.random()':
                    precision = float(params[ind + 1])
                    precision_ind = ind + 1
                else:
                    if param != '' and 'random()' not in param and ind != precision_ind:
                        to_extend = param.split(' and ')
                        if '' in to_extend:
                            to_extend.remove('')
                        initial_index_arguments.extend(to_extend)
        else:
            initial_index_arguments = function_arguments.split(' and ')
        stripped_index_arguments = []
        for value in initial_index_arguments:
            stripped_index_arguments.append(value.replace(" ", ""))
        # for dict_name, props in iterable._persistent_props.iteritems():
        index_arguments = set()
        non_index_arguments = []
        # if 'indexed_values' in props:

        for value in stripped_index_arguments:
            to_append = str(value)
            if '<' in value:
                splitval = value.split('<')
                for pos in splitval:
                    if pos not in indexed_args:
                        try:
                            newval = eval(pos)
                            to_append = value.replace(str(pos), str(newval))
                        except Exception as e:
                            for tup in vars_to_vals:
                                if tup[0] == pos:
                                    to_append = value.replace(str(pos), str(tup[1]))
                if splitval[0] in indexed_args or splitval[1] in indexed_args:
                    index_arguments.add(to_append)
                else:
                    non_index_arguments.append(to_append)
            elif '>' in value:
                splitval = value.split('>')
                for pos in splitval:
                    if pos not in indexed_args:
                        try:
                            newval = eval(pos)
                            to_append = value.replace(str(pos), str(newval))
                        except Exception as e:
                            for tup in vars_to_vals:
                                if tup[0] == pos:
                                    to_append = value.replace(str(pos), str(tup[1]))
                if splitval[0] in indexed_args or splitval[1] in indexed_args:
                    index_arguments.add(to_append)
                else:
                    non_index_arguments.append(to_append)
            else:
                non_index_arguments.append(value)

        non_index_arguments = ' and '.join(non_index_arguments)

        min_arguments = {}
        max_arguments = {}
        for argument in index_arguments:
            if '<' in str(argument):
                splitarg = (str(argument).replace(' ', '')).split('<')
                if len(splitarg) == 2:
                    val = str(splitarg[0])
                    max_arguments[val] = float(splitarg[1])
                elif len(splitarg) == 3:
                    val = str(splitarg[1])
                    min_arguments[val] = float(splitarg[0])
                    max_arguments[val] = float(splitarg[2])
                else:
                    log.error("Malformed filtering predicate:" + str(argument))
            if '>' in str(argument):
                splitarg = (str(argument).replace(' ', '')).split('>')
                if len(splitarg) == 2:
                    val = str(splitarg[0])
                    min_arguments[val] = float(splitarg[1])
                elif len(splitarg) == 3:
                    val = str(splitarg[1])
                    min_arguments[val] = float(splitarg[2])
                    max_arguments[val] = float(splitarg[0])
                else:
                    log.error("Malformed filtering predicate:" + str(argument))
        from_p = []
        to_p = []
        for indexed_element in indexed_args:
            from_p.append(min_arguments[indexed_element])
            to_p.append(max_arguments[indexed_element])
        from qbeast import QbeastMeta, QbeastIterator
        qmeta = QbeastMeta(
            non_index_arguments,
            from_p, to_p,
            precision)
        it = QbeastIterator(father._primary_keys, father._columns,
                            father._ksp + "." + father._table,
                            qmeta)
        return it
    else:
        filtered = python_filter(lambda_filter, iterable)
        return filtered


if not filter == hecuba_filter:
    python_filter = filter
    filter = hecuba_filter

global config
config = Config()
