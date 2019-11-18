#!/bin/bash
###############################################################################################################
#													      #
#                                 Application Node Launcher for Slurm clusters                                #
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
APP_PATH=$(cat $C4S_HOME/app-"$UNIQ_ID".txt)

if [[ "0$SCHEMA" != "0" ]]; then
  node1=$(echo $CONTACT_NAMES | awk -F ',' '{print $1}')

  echo "Connecting to $node1 for tables creation. Schema $SCHEMA."
  cqlsh $node1 -f $SCHEMA
  sleep 10
fi

echo "Launching application: $APP_PATH"
eval $APP_PATH
echo "[INFO] The application execution has stopped in node $(hostname)"
