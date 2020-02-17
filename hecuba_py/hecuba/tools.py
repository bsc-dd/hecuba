import uuid
import typing
from collections import OrderedDict
from hecuba.IStorage import IStorage
from storage.cql_iface import config

def storage_id_from_name(name):
    return uuid.uuid3(uuid.NAMESPACE_DNS, name)


def process_path(module_path):
    """
    Method to obtain module and class_name from a module path
    Args:
        module_path(String): path in the format module.class_name
    Returns:
        tuple containing class_name and module
    """

    if module_path == 'numpy.ndarray':
        return 'StorageNumpy', 'hecuba.hnumpy'
    if module_path == 'StorageDict':
        return 'StorageDict', 'hecuba.hdict'
    last = 0
    for key, i in enumerate(module_path):
        if i == '.' and key > last:
            last = key
    module = module_path[:last]
    class_name = module_path[last + 1:]
    return class_name, module


"""
 Cassandra related methods
"""

_size_estimates = config.session.prepare(("SELECT mean_partition_size, partitions_count "
                                          "FROM system.size_estimates WHERE keyspace_name=? and table_name=?"))
_max_token = int(((2 ** 63) - 1))  # type: int
_min_token = int(-2 ** 63)  # type: int

_select_istorage_meta = config.session.prepare("SELECT * FROM hecuba.istorage WHERE storage_id = ?")


def extract_ks_tab(name):
    """
    Method used to obtain keyspace and table from a given name
    Args:
        name: a string containing keyspace name and table name, or only table name
    Returns:
        a tuple containing keyspace name and table name
    """
    if not name:
        return None, None

    sp = name.split(".")
    if len(sp) == 2:
        ksp = sp[0]
        table = sp[1]
    else:
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


def generate_token_ring_ranges():
    ring = config.cluster.metadata.token_map.ring
    tokens = [token.value for token in ring]
    return discrete_token_ranges(tokens)


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


def build_remotely(args):
    """
    Takes the information which consists of at least the type,
    :raises TypeError if the object class doesn't subclass IStorage
    :param obj_info: Contains the information to be used to create the IStorage obj
    :return: An IStorage object
    """
    if "built_remotely" not in args.keys():
        built_remotely = True
    else:
        built_remotely = args["built_remotely"]

    obj_type = args.get('class_name', args.get('type', None))
    if obj_type is None:
        raise TypeError("Trying to build an IStorage obj without giving the type")

    imported_class = obj_type

    args = {k: v for k, v in args.items() if k in imported_class.args_names}
    args.pop('class_name', None)
    args["built_remotely"] = built_remotely

    return imported_class(**args)


def import_class(module_path):
    """
    Method to obtain module and class_name from a module path
    Args:
        module_path(String): path in the format module.class_name
    Returns:
        tuple containing class_name and module
    """
    class_name, mod = process_path(module_path)

    try:
        mod = __import__(mod, globals(), locals(), [class_name], 0)
    except ValueError:
        raise ValueError("Can't import class {} from module {}".format(class_name, mod))

    imported_class = getattr(mod, class_name)
    return imported_class


def build_data_model(description):
    res = {}
    for k, v in description.items():
        dt = update_type(v["type"])
        try:
            keys = build_data_model(v["primary_keys"])
            values = build_data_model(v["columns"])
            res[k] = {"keys": keys, "cols": values, "type": dt}
        except KeyError:
            res[k] = dt
    return res
    # {k: update_type(v['type']) for k, v in persistent_props.items()}


def update_type(d):
    if d == 'text':
        return str
    res = import_class(d)
    return res


def transform_to_dm(ob, depth=0):
    """

    :param ob:
    :return: List or dict
    """
    if issubclass(ob, IStorage) and depth>0:
        return ob
    elif issubclass(ob, typing.Dict):
        fields = {}

        keys = transform_to_dm(ob.__args__[0], depth+1)  # Keys
        cols = transform_to_dm(ob.__args__[1], depth+1)  # Cols

        if isinstance(keys, list):
            keys = {"key{}".format(i): transform_to_dm(v, depth+1) for i, v in enumerate(keys)}
        if isinstance(cols, list):
            cols = {"col{}".format(i): transform_to_dm(v,depth+1) for i, v in enumerate(cols)}

        fields["value_id"] = keys
        fields["cols"] = cols
        return fields

    elif hasattr(ob, '__annotations__'):
        annot = ob.__annotations__
        if isinstance(annot, OrderedDict):
            return {k: transform_to_dm(v, depth+1) for k,v in annot.items()}
        elif isinstance(annot, dict):
            return {k: transform_to_dm(v, depth+1) for k,v in annot.items()}
        else:
            raise NotImplemented

    elif hasattr(ob, '__args__'):
        if issubclass(ob, typing.Tuple):
            t = [transform_to_dm(cl, depth+1) for cl in ob.__args__ if cl != ()]
            return tuple(t)
        return [transform_to_dm(cl, depth+1) for cl in ob.__args__ if cl != ()]
    return ob
