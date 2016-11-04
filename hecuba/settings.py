def init():
    if 'apppath' not in globals():
        import os
        global apppath, contact_names, nodePort
        apppath = os.environ['APPPATH']
        try:
            nodePort = os.environ['NODEPORT']
        except:
            print 'using default nodePort 9042'
            nodePort = 9042

        try:
            contact_names = os.environ['CONTACT_NAMES'].split(",")
        except:
            print 'using default contact point localhost'
            contact_names = ['localhost']



    if 'cluster' not in globals():
        global cluster, session
        from cassandra.cluster import Cluster
        cluster = Cluster(contact_points=contact_names, port=nodePort)
        session = cluster.connect()


init()
