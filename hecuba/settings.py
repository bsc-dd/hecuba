def init():
    if 'apppath' not in globals():
        import os
        global apppath, contact_names, nodePort
        try:
            apppath = os.environ['APPPATH']
        except KeyError:
            apppath = os.getcwd()
    try:
        nodePort = os.environ['NODEPORT']
    except KeyError:
        print 'using default nodePort 9042'
        nodePort = 9042

    try:
        contact_names = os.environ['CONTACT_NAMES'].split(",")
    except KeyError:
        print 'using default contact point localhost'
        contact_names = ['localhost']
    if 'cluster' not in globals():
        global cluster, session
        if 'TESTING' in os.environ:
            class clusterMock: pass
            class sessionMock: pass
            cluster = clusterMock()
            session = sessionMock()
        else:
            from cassandra.cluster import Cluster
            print 'CREATING GLOBAL SESSION'
            cluster = Cluster(contact_points=contact_names, port=nodePort)
            session = cluster.connect()
            session.execute(
                "CREATE KEYSPACE IF NOT EXISTS hecuba WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3 }")
            session.execute(
                'CREATE TABLE IF NOT EXISTS hecuba.blocks (blockid text, block_classname text,storageobj_classname text, tkns list<bigint>, ' +
                'entry_point text , port int, ksp text , tab text , dict_name text , obj_type text, PRIMARY KEY(blockid))')

    import logging
    logging.basicConfig()


init()
