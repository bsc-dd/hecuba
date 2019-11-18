#!/bin/bash
###############################################################################################################
#													      #
#                                  Cassandra Node Launcher for Slurm clusters                                 #
#                                          Eloy Gil - eloy.gil@bsc.es                                         #
#													      #
#                                     Barcelona Supercomputing Center 2018                                    #
#		                                     .-.--_                                       	      #
#                    			           ,´,´.´   `.                                     	      #
#              			                   | | | BSC |                                     	      #
#                   			           `.`.`. _ .´                                     	      #
#                        		             `·`··                                         	      #
#													      #
###############################################################################################################

export C4S_HOME=$HOME/.c4s
UNIQ_ID=${1}
if [ "$(cat $C4S_HOME/casslist-"$UNIQ_ID".txt | grep $(hostname))" != "" ]; then
    echo "JAVA_HOME="$JAVA_HOME
    echo "Cassandra node $(hostname) is starting now..."
    echo "CASS_HOME="$CASS_HOME
    $CASS_HOME/bin/cassandra -Dcassandra.config=file://$C4S_HOME/conf/cassandra-$(hostname).yaml -f | awk "{ print  $(hostname),\$0 }"
    echo "Cassandra has stopped in node $(hostname)"
else
    echo "Node $(hostname) is not part of the Cassandra node list."
fi
