class Config: pass

def init():
    import logging
    logging.basicConfig()
    global config
    config = Config()
    if 'apppath' not in globals():
        import os
        try:
            config.apppath = os.environ['APPPATH']
            logging.info('APPPATH: %s', config.apppath)
        except KeyError:
            config.apppath = os.getcwd()
            logging.warn('APPPATH missing, using default value %s', config.apppath)
    try:
        config.nodePort = os.environ['NODEPORT']
        logging.info('NODEPORT: %s', config.nodePort)
    except KeyError:
        logging.warn('using default nodePort 9042')
        config.nodePort = 9042

    try:
        config.contact_names = os.environ['CONTACT_NAMES'].split(",")
        logging.info('CONTACT_NAMES: %s', str.join(" ", config.contact_names))
    except KeyError:
        logging.warn('using default contact point localhost')
        config.contact_names = ['localhost']
    if 'cluster' not in globals():
        global cluster, session
        if 'TESTING' in os.environ:
            class clusterMock: pass
            class sessionMock: pass

            cluster = clusterMock()
            session = sessionMock()
            logging.info('TESTING ENVIRONMENT: connection to %s', config.apppath)

        else:
            from cassandra.cluster import Cluster
            logging.info('Initializing global session')
            cluster = Cluster(contact_points=config.contact_names, port=config.nodePort)
            session = cluster.connect()
            session.execute(
                "CREATE KEYSPACE IF NOT EXISTS hecuba WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3 }")
            session.execute(
                'CREATE TABLE IF NOT EXISTS hecuba.blocks (blockid text, block_classname text,storageobj_classname text, tkns list<bigint>, ' +
                'entry_point text , port int, ksp text , tab text , dict_name text , obj_type text, PRIMARY KEY(blockid))')


    config.cache_activated = True
    config.batch_size = 100
    config.max_cache_size = 100


init()
