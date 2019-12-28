import sys
from cassandra.cluster import Cluster, NoHostAvailable
cluster = Cluster(contact_points=[sys.argv[-1]], port=9042)
try:
    session = cluster.connect()
    exit_code = 0
except NoHostAvailable:
    exit_code = 1

cluster.shutdown()
sys.exit(exit_code)
