import uuid
import typing
from collections import OrderedDict
import hecuba


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

    res = module_path.split('.')
    if len(res) == 1:
        mod = "builtins"
        class_name = module_path
    else:
        mod = res[0]
        class_name = module_path[len(mod) + 1:]

    return class_name, mod


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
    if issubclass(ob, hecuba.IStorage.IStorage) and depth>0:
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
