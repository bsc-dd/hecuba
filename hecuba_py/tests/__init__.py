import atexit
import ccmlib.cluster
import os
import sys
import tempfile
import logging
from distutils.util import strtobool


class TestConfig:
    pass


test_config = TestConfig()
test_config.n_nodes = int(os.environ.get('TEST_CASSANDRA_N_NODES', '2'))
TEST_DEBUG = strtobool(os.environ.get("TEST_DEBUG", "False").lower())
if TEST_DEBUG:
    logging.warning(("You are using TEST_DEBUG=True, a Cassandra cluster must be already running. "
                     "Keep in mind that the results of the test might be altered by data already existing."))


def set_ccm_cluster():
    global test_config
    test_config.ccm_cluster = ccmlib.cluster.Cluster(
        tempfile.mkdtemp("tmp_data"),
        'hecuba_test',
        cassandra_version=os.environ.get('TEST_CASSANDRA_VERSION', '3.11.4'))


def set_up_default_cassandra():
    global test_config
    if hasattr(test_config, "ccm_cluster") and any(
            map(lambda node: node.is_live(), test_config.ccm_cluster.nodes.values())):
        return

    set_ccm_cluster()
    try:
        test_config.ccm_cluster.populate(test_config.n_nodes).start(allow_root=True,jvm_args=["-Xss512k"])
    except Exception as a:
        if TEST_DEBUG:
            logging.warning("TEST_DEBUG: ignoring exception")
        else:
            raise a

    if 'hecuba' in sys.modules:
        import importlib
        import hecuba
        importlib.reload(hecuba)


@atexit.register
def turning_down_cassandra():
    global test_config
    if test_config is None or not hasattr(test_config, "ccm_cluster"):
        return

    print("Turning down Cassandra")
    from hfetch import disconnectCassandra
    disconnectCassandra()
    test_config.ccm_cluster.stop()
    test_config.ccm_cluster.clear()


set_up_default_cassandra()
