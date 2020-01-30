from .storage_iface import StorageIface


def select_storage_api():
    from .cql_iface.cql_iface import CQLIface
    return CQLIface()


StorageAPI = select_storage_api()