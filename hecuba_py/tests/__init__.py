import atexit
import ccmlib.cluster
import os
import sys
import tempfile
import logging
from distutils.util import strtobool


class TestConfig:
    n_nodes = int(os.environ.get('TEST_CASSANDRA_N_NODES', '2'))
    ccm_cluster = None


TEST_DEBUG = strtobool(os.environ.get("TEST_DEBUG", "False").lower())
if TEST_DEBUG:
    logging.warning(("You are using TEST_DEBUG=True, a Cassandra cluster must be already running. "
                     "Keep in mind that the results of the test might be altered by data already existing."))


def set_ccm_cluster():
    TestConfig.ccm_cluster = ccmlib.cluster.Cluster(
        tempfile.mkdtemp("tmp_data"),
        'hecuba_test',
        cassandra_version=os.environ.get('TEST_CASSANDRA_VERSION', '3.11.4'))


def set_up_default_cassandra():
    if TestConfig.ccm_cluster and any(map(lambda node: node.is_live(), TestConfig.ccm_cluster.nodes.values())):
        return

    set_ccm_cluster()
    try:
        TestConfig.ccm_cluster.populate(TestConfig.n_nodes).start(allow_root=True)
    except Exception as ex:
        if TEST_DEBUG:
            logging.warning("TEST_DEBUG: ignoring exception")
        else:
            raise ex

    if 'hecuba' in sys.modules:
        import importlib
        import hecuba
        importlib.reload(hecuba)


@atexit.register
def turning_down_cassandra():
    if TestConfig is None or TestConfig.ccm_cluster is None:
        return

    print("Shutting down Cassandra")
    from hfetch import disconnectCassandra
    disconnectCassandra()
    TestConfig.ccm_cluster.stop()
    TestConfig.ccm_cluster.clear()
    TestConfig.ccm_cluster = None


set_up_default_cassandra()
