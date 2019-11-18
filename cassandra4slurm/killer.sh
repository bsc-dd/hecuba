#!/bin/bash
###############################################################################################################
#                                                                                                             #
#                                           Cassandra Process Killer                                          #
#                                          Eloy Gil - eloy.gil@bsc.es                                         #
#                                                                                                             #
#                                     Barcelona Supercomputing Center 2018                                    #
#                                                    .-.--_                                                   #
#                                                  ,´,´.´   `.                                                #
#                                                  | | | BSC |                                                #
#                                                  `.`.`. _ .´                                                #
#                                                    `·`··                                                    #
#                                                                                                             #
###############################################################################################################

PID=$(ps aux | grep java | grep cassandra | awk '{ print $2 }')
if [ "$PID" != "" ]; then
    echo "Host $(hostname) is killing its cassandra process with PID: "$PID
    kill $PID
else
    echo "Host $(hostname) can not find a running cassandra process to kill."
fi
exit
