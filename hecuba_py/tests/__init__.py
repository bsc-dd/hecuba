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
    logging.warning(("You are using TEST_DEBUG=True. Remember to kill and clean the CCM cluster and keep in mind that the "
                 "results of the test might be altered."))


def set_ccm_cluster():
    test_config.ccm_cluster = ccmlib.cluster.Cluster(
        tempfile.mkdtemp("tmp_data"),
        'hecuba_test',
        cassandra_version=os.environ.get('TEST_CASSANDRA_VERSION', '3.11.4'))


def set_up_default_cassandra():
    set_ccm_cluster()
    try:
        test_config.ccm_cluster.populate(test_config.n_nodes).start(allow_root=True,jvm_args=["-Xss512k"])
    except Exception as a:
        if TEST_DEBUG:
            logging.warning("TEST_DEBUG: ignoring exception")
        else:
            raise a

    if 'hecuba' in sys.modules:
        reload(hecuba)


@atexit.register
def turning_down_cassandra():
    if TEST_DEBUG:
        print("Leaving Cassandra running")
    else:
        print("Turning down Cassandra")
        if test_config is not None and test_config.ccm_cluster is not None:
            from hfetch import disconnectCassandra
            disconnectCassandra()
            test_config.ccm_cluster.stop()
            test_config.ccm_cluster.clear()


set_up_default_cassandra()
