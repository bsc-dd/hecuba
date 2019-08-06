import uuid

valid_types = ['counter', 'text', 'boolean', 'decimal', 'double', 'int', 'list', 'set', 'map', 'bigint', 'blob',
               'tuple', 'dict', 'float', 'numpy.ndarray']

basic_types = valid_types[:-1]


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

    # Import the class defined by obj_type
    cname, module = process_path(obj_type)

    '''
    if obj_type == str(StorageNumpy.__class__):
        return StorageNumpy(name=args["name"], storage_id=args["storage_id"])
    '''
    try:
        mod = __import__(module, globals(), locals(), [cname], 0)
    except ValueError:
        raise ValueError("Can't import class {} from module {}".format(cname, module))

    imported_class = getattr(mod, cname)

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
