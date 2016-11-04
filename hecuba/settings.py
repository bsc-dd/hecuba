def init():
    if 'apppath' not in globals():
        import os
        global apppath
        apppath = os.environ['APPPATH'] + '/conf/storage_params.txt'
        try:
            file = open(apppath, 'r')

            for line in file:
                exec line
        except:
            print 'Impossible to load storage_params.'

    if 'cluster' not in globals():
        global cluster, session
        from cassandra.cluster import Cluster
        cluster = Cluster(contact_points=contact_names, port=nodePort)
        session = cluster.connect()


init()
