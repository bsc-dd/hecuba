#!/bin/bash
###############################################################################################################
#													      #
#                                        Cassandra Node Launcher for HPC                                      #
#                                          Eloy Gil - eloy.gil@bsc.es                                         #
#													      #
#                                        Barcelona Supercomputing Center                                      #
#		                                     .-.--_                                       	      #
#                    			           ,´,´.´   `.                                     	      #
#              			                   | | | BSC |                                     	      #
#                   			           `.`.`. _ .´                                     	      #
#                        		             `·`··                                         	      #
#													      #
###############################################################################################################

export C4S_HOME=$HOME/.c4s
UNIQ_ID=${1}
CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg

source $HECUBA_ROOT/bin/cassandra4slurm/hecuba_debug.sh
source $CFG_FILE    # To get CASSANDRA_LOG_DIR
# Aggregate all logs in a single UNIQ_ID directory
[ ! -z "$CASSANDRA_LOG_DIR" ] \
    && export CASSANDRA_LOG_DIR="$CASSANDRA_LOG_DIR/$UNIQ_ID"

HOSTNAMEIP=$(get_node_ip $(hostname) $CASS_IFACE)

if [ "$(cat $C4S_HOME/casslist-"$UNIQ_ID".txt.ips | grep $HOSTNAMEIP)" != "" ]; then
    INDEX=$(awk "/$HOSTNAMEIP/{ print NR; exit }" $C4S_HOME/casslist-"$UNIQ_ID".txt.ips)
    #SLEEP_TIME=$(((INDEX - 7) * 8))
    SLEEP_TIME=30
    if [ "$INDEX" -gt 2 ]; then
        echo "Node "$(hostname)" will sleep for "$SLEEP_TIME" seconds."
        sleep $SLEEP_TIME
    fi

    DBG " JAVA_HOME="$JAVA_HOME
    DBG " Cassandra node $(hostname) is starting now..."
    DBG " CASS_HOME="$CASS_HOME
    DBG " CASSANDDRA CONF $C4S_HOME/conf/${UNIQ_ID}/cassandra-${HOSTNAMEIP}.yaml"
    $CASS_HOME/bin/cassandra -Dcassandra.consistent.rangemovement=false -Dcassandra.config=file://$C4S_HOME/conf/${UNIQ_ID}/cassandra-${HOSTNAMEIP}.yaml -f  | awk "{ print  \""$HOSTNAMEIP"\",\$0 }"
    echo "Cassandra has stopped in node $(hostname)"
else
    echo "Node $(hostname) is not part of the Cassandra node list."
fi
