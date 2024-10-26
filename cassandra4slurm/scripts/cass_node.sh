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
CFG_FILE=$C4S_HOME/conf/${UNIQ_ID}/cassandra4slurm.cfg

MODULE_PATH=$HECUBA_ROOT/bin/cassandra4slurm

source $HECUBA_ROOT/bin/cassandra4slurm/hecuba_debug.sh
source $CFG_FILE    # To get CASSANDRA_LOG_DIR
# Aggregate all logs in a single UNIQ_ID directory
[ ! -z "$CASSANDRA_LOG_DIR" ] \
    && export CASSANDRA_LOG_DIR="$CASSANDRA_LOG_DIR/$UNIQ_ID"

function launch_arrow_helper () {
    ! is_HECUBA_ARROW_enabled  && return

    # Launch the 'arrow_helper' tool at each node in NODES, and leave their logs in LOGDIR
    NODES=$C4S_HOME/casslist-"$UNIQ_ID".txt
    LOGDIR=$CASSANDRA_LOG_DIR/arrow
    if [ ! -d $LOGDIR ]; then
        DBG " Creating directory to store Arrow helper logs at [$LOGDIR]:"
        mkdir -p $LOGDIR
    fi

    ARROW_HELPER=$MODULE_PATH/launch_arrow_helper.sh

    DBG " Launching Arrow helper at [$i] Log at [$LOGDIR/arrow_helper.$i.out]:"
    HECUBA_ROOT=$HECUBA_ROOT $ARROW_HELPER $UNIQ_ID $LOGDIR/arrow_helper.$(hostname).out &
}

HOSTNAMEIP=$(get_node_ip $(hostname) $CASS_IFACE)

if [ "$(cat $C4S_HOME/casslist-"$UNIQ_ID".txt.ips | grep $HOSTNAMEIP)" != "" ]; then

    launch_arrow_helper

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
    export CASSPIDFILE=$C4S_HOME/conf/${UNIQ_ID}/cassandra-${HOSTNAMEIP}.pid
    $CASS_HOME/bin/cassandra \
	    -Dcassandra.consistent.rangemovement=false \
	    -Dcassandra.config=file://$C4S_HOME/conf/${UNIQ_ID}/cassandra-${HOSTNAMEIP}.yaml \
	    -p $CASSPIDFILE \
	    #-f  \
	    #| awk "{ print  \""$HOSTNAMEIP"\",\$0 }"
    # Wait for cassandra to start up and write the PID file
    while [ ! -s $CASSPIDFILE ]; do
    	echo "Waiting Cassandra writing PID @$(hostname)"
	sleep 1
    done

    # Launch cassandra manager
    echo "Starting Cassandra manager for PID $(cat $CASSPIDFILE)"
    $HECUBA_ROOT/bin/cass_mgr $(cat $CASSPIDFILE) &

    # Wait for termination --> 'wait' does not work :(
    while [ -f $CASSPIDFILE ]; do
	sleep 1
    done


    echo "Cassandra has stopped in node $(hostname)"

else
    echo "Node $(hostname) is not part of the Cassandra node list."
fi
