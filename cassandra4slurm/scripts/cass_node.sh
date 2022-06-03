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
source $HECUBA_ROOT/bin/cassandra4slurm/hecuba_debug.sh


if [ "$(cat $C4S_HOME/casslist-"$UNIQ_ID".txt | grep $(hostname))" != "" ]; then
    INDEX=$(awk "/$(hostname)/{ print NR; exit }" $C4S_HOME/casslist-"$UNIQ_ID".txt)
    #SLEEP_TIME=$(((INDEX - 7) * 8))
    SLEEP_TIME=30
    if [ "$INDEX" -gt 2 ]; then
        echo "Node "$(hostname)" will sleep for "$SLEEP_TIME" seconds."
        sleep $SLEEP_TIME
    fi

    DBG " JAVA_HOME="$JAVA_HOME
    DBG " Cassandra node $(hostname) is starting now..."
    DBG " CASS_HOME="$CASS_HOME
    $CASS_HOME/bin/cassandra -Dcassandra.consistent.rangemovement=false -Dcassandra.config=file://$C4S_HOME/conf/cassandra-$(hostname).yaml -f  | awk "{ print  \""$(hostname)"\",\$0 }"
    echo "Cassandra has stopped in node $(hostname)"
else
    echo "Node $(hostname) is not part of the Cassandra node list."
fi
