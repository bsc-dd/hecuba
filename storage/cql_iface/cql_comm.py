from hecuba import log
from hfetch import Hcache
from . import config
import uuid
from hecuba.parser import _conversions

"""
 Cassandra related methods
"""

_size_estimates = config.session.prepare(("SELECT mean_partition_size, partitions_count "
                                          "FROM system.size_estimates WHERE keyspace_name=? and table_name=?"))
_max_token = int(((2 ** 63) - 1))  # type: int
_min_token = int(-2 ** 63)  # type: int

_select_istorage_meta = config.session.prepare("SELECT * FROM hecuba.istorage WHERE storage_id = ?")


def extract_ksp_table(name):
    """
    Method used to obtain keyspace and table from a given name
    Args:
        name: a string containing keyspace name and table name, or only table name
    Returns:
        a tuple containing keyspace name and table name
    """

    try:
        ksp = name[:name.index('.')]
        table = name[len(ksp) + 1:]
    except ValueError as ex:
        ksp = config.execution_name
        table = name
    return ksp.lower(), table.lower()


def tokens_partitions(ksp, table, tokens_ranges):
    """
    Method that calculates the new token partitions for a given object
    Args:
        tokens: current number of tokens of the object
        min_tokens_per_worker: defined minimum number of tokens
        number_of_workers: defined
    Returns:
        a partition every time it's called
        :type tokens_ranges: list[(long,long)]
    """
    from collections import defaultdict
    from bisect import bisect_right
    from cassandra.metadata import Murmur3Token

    splits_per_node = config.splits_per_node
    token_range_size = config.token_range_size
    target_token_range_size = config.target_token_range_size

    tm = config.cluster.metadata.token_map
    tmap = tm.tokens_to_hosts_by_ks.get(ksp, None)

    tokens_murmur3 = map(lambda a: (Murmur3Token(a[0]), a[1]), tokens_ranges)
    if not tmap:
        tm.rebuild_keyspace(ksp, build_if_absent=True)
        tmap = tm.tokens_to_hosts_by_ks[ksp]

    tokens_per_node = defaultdict(list)
    for tmumur, t_to in tokens_murmur3:
        point = bisect_right(tm.ring, tmumur)
        if point == len(tm.ring):
            tokens_per_node[tmap[tm.ring[0]][0]].append((tmumur.value, t_to))
        else:
            tokens_per_node[tmap[tm.ring[point]][0]].append((tmumur.value, t_to))

    n_nodes = len(tokens_per_node)
    step_size = _max_token // (splits_per_node * n_nodes)
    if token_range_size:
        step_size = token_range_size
    elif target_token_range_size:
        one = config.session.execute(_size_estimates, [ksp, table]).one()
        if one:
            (mean_p_size, p_count) = one
            estimated_size = mean_p_size * p_count
            if estimated_size > 0:
                step_size = _max_token // (
                    max(estimated_size / target_token_range_size,
                        splits_per_node * n_nodes)
                )

    for tokens_in_node in tokens_per_node.values():
        partition = []
        for fraction, to in tokens_in_node:
            while fraction < to - step_size:
                partition.append((fraction, fraction + step_size))
                fraction += step_size
            partition.append((fraction, to))
        group_size = max(len(partition) // splits_per_node, 1)
        for i in range(0, len(partition), group_size):
            yield partition[i:i + group_size]


def discrete_token_ranges(tokens):
    """
    Makes proper tokens ranges ensuring that in a tuple (a,b) a <= b
    Args:
        tokens:  a list of tokens [1, 0, 10]
    Returns:
         a rationalized list [(-1, 0),(0,10),(10, max)]
    """
    tokens.sort()
    if len(tokens) == 0:
        return tokens
    if tokens[0] > _min_token:
        token_ranges = [(_min_token, tokens[0])]
    else:
        token_ranges = []
    n_tns = len(tokens)
    for i in range(0, n_tns - 1):
        token_ranges.append((tokens[i], tokens[i + 1]))
    token_ranges.append((tokens[n_tns - 1], _max_token))
    return token_ranges


def count_name_collision(ksp, table, attribute):
    import re
    m = re.compile("^%s_%s(_[0-9]+)?$" % (table, attribute))
    q = config.session.execute("SELECT table_name FROM  system_schema.tables WHERE keyspace_name = %s",
                               [ksp])
    return sum(1 for elem in q if m.match(elem[0]))


def get_istorage_attrs(storage_id):
    return list(config.session.execute(_select_istorage_meta, [storage_id]))


class CqlCOMM(object):
    istorage_prepared_st = config.session.prepare('INSERT INTO hecuba.istorage'
                                                  '(storage_id, name, data_model)'
                                                  'VALUES (?,?,?)')

    istorage_remove_entry = config.session.prepare('DELETE FROM hecuba.istorage WHERE storage_id = ?')
    istorage_read_entry = config.session.prepare('SELECT * FROM hecuba.istorage WHERE storage_id = ?')

    @staticmethod
    def register_istorage(obj_id, obj_name, data_model):
        obj_info = [obj_id, obj_name, str(data_model)]
        config.execute(CqlCOMM.istorage_prepared_st, obj_info)

    @staticmethod
    def register_data_model(data_model_id, definition):
        # extract keys, values and so on
        pass

    @staticmethod
    def delete_data(object_id):
        res = config.execute(CqlCOMM.istorage_read_entry, [object_id])
        if res:
            res = res.one()
        config.execute(CqlCOMM.istorage_remove_entry, [object_id])
        # TODO Use res to delete the appropriate data, maybe async

    @staticmethod
    def create_table(object_id, name, definition):
        # StorageObj for now
        ksp, table = extract_ksp_table(name)
        query_keyspace = "CREATE KEYSPACE IF NOT EXISTS %s WITH replication = %s" % (ksp, config.replication)
        config.session.execute(query_keyspace)

        primary_keys = definition['keys']
        columns = definition['cols']

        if not primary_keys:
            primary_keys = {"storage_id": uuid.UUID}

        all_keys = ",".join("%s %s" % (k, _conversions[v.__name__]) for k, v in primary_keys.items())

        all_cols = ",".join("%s %s" % (k, _conversions[v.__name__]) for k, v in columns.items())

        if all_cols:
            total_cols = all_keys + ',' + all_cols
        else:
            total_cols = all_keys

        query_table = "CREATE TABLE IF NOT EXISTS %s.%s (%s, PRIMARY KEY (%s));" \
                      % (ksp,
                         table,
                         total_cols,
                         str.join(',', primary_keys.keys()))
        try:
            log.debug('MAKE PERSISTENCE: %s', query_table)
            config.session.execute(query_table)
        except Exception as ex:
            log.warn("Error creating the StorageDict table: %s %s", query_table, ex)
            raise ex

    @staticmethod
    def create_hcache(object_id, name, definition):
        ksp, table = extract_ksp_table(name)

        if definition["keys"]:
            keys = [k for k in definition["keys"].keys()]
            columns = [k for k in definition["cols"].keys()]

            hcache_params = (ksp, table, object_id, [(-2 ** 63, 2 ** 63 - 1)], keys, columns,
                             {'cache_size': config.max_cache_size,
                              'writer_par': config.write_callbacks_number,
                              'writer_buffer': config.write_buffer_size,
                              'timestamped_writes': config.timestamped_writes})
            return Hcache(*hcache_params)

        else:

            class HcacheWrapper(object):
                def __init__(self, attributes, object_id, ksp, table):
                    self.internal_caches = {}
                    self.object_id = object_id
                    for attr in attributes:
                        hc = Hcache(ksp, table, object_id, [(-2 ** 63, 2 ** 63 - 1)], ["storage_id"], [attr],
                                    {'cache_size': config.max_cache_size,
                                     'writer_par': config.write_callbacks_number,
                                     'writer_buffer': config.write_buffer_size,
                                     'timestamped_writes': config.timestamped_writes})

                        self.internal_caches[attr] = hc

                def get_row(self, attr):
                    return self.internal_caches[attr].get_row([self.object_id])[0]

                def put_row(self, attr, val):
                    self.internal_caches[attr].put_row([self.object_id], [val])

            return HcacheWrapper(definition["cols"].keys(), object_id, ksp, table)
