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

# Do NOT use hostname directly! There are machines that return a differnt name than the one used in SLURM.
HOSTNAMEIP=$(get_node_ip $(hostname) $CASS_IFACE)

# Aggregate all logs in a single UNIQ_ID directory
[ -z "$LOG_PATH" ] \
    && echo "ERROR: LOG_PATH not defined. Using default" \
    && export LOG_PATH=$HOME/.c4s/logs

[ -z "$CASSANDRA_LOG_DIR" ] \
    && echo "WARNING: CASSANDRA_LOG_DIR not defined. Using default" \
    && export CASSANDRA_LOG_DIR=$LOG_PATH

export CASSANDRA_LOG_DIR="$CASSANDRA_LOG_DIR/$UNIQ_ID"

DBG "enqueue_cass_node.sh: CASSANDRA_LOG_DIR=$CASSANDRA_LOG_DIR"

if [ "$(cat $C4S_HOME/casslist-"$UNIQ_ID".txt.ips | grep $HOSTNAMEIP)" != "" ]; then
    INDEX=$(awk "/$HOSTNAMEIP/{ print NR; exit }" $C4S_HOME/casslist-"$UNIQ_ID".txt.ips)
    #SLEEP_TIME=$(((INDEX - 7) * 8))
    SLEEP_TIME=30
    if [ "$INDEX" -gt 2 ]; then
        echo "Node "$(hostname)" will sleep for "$SLEEP_TIME" seconds."
        sleep $SLEEP_TIME
    fi
    ENVIRON_TO_LOAD=$C4S_HOME/environ-"$UNIQ_ID".txt
    if [ -f $ENVIRON_TO_LOAD ]; then
        DBG " ENVIRON_TO_LOAD=$ENVIRON_TO_LOAD"
        source $ENVIRON_TO_LOAD
    fi
    DBG " JAVA_HOME="$JAVA_HOME
    DBG " Cassandra node $(hostname) is starting now..."
    DBG " CASS_HOME="$CASS_HOME
    # Launch cassandra and prefix the output with 'hostname'
    $CASS_HOME/bin/cassandra -Dcassandra.consistent.rangemovement=false -Dcassandra.config=file://$C4S_HOME/conf/${UNIQ_ID}/cassandra-$HOSTNAMEIP.yaml -f | awk "{ print  \"$HOSTNAMEIP\",\$0 }"
    echo "Cassandra has stopped in node $(hostname)"
else
    echo "Node $(hostname) is not part of the Cassandra node list."
fi
