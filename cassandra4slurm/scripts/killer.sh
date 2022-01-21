#!/bin/bash
###############################################################################################################
#                                                                                                             #
#                                           Cassandra Process Killer                                          #
#                                          Eloy Gil - eloy.gil@bsc.es                                         #
#                                                                                                             #
#                                        Barcelona Supercomputing Center                                      #
#                                                    .-.--_                                                   #
#                                                  ,´,´.´   `.                                                #
#                                                  | | | BSC |                                                #
#                                                  `.`.`. _ .´                                                #
#                                                    `·`··                                                    #
#                                                                                                             #
###############################################################################################################

# Kill Cassandra instance
PID=$(ps aux | grep java | grep cassandra | awk '{ print $2 }')
if [ "$PID" != "" ]; then
    echo "Host $(hostname) is killing its cassandra process with PID: "$PID
    kill $PID
else
    echo "Host $(hostname) can not find a running cassandra process to kill."
fi

# Kill Arrow Helpers
PID=$(ps aux |grep arrow_helper | awk '{print $2}')
if [ "$PID" != "" ]; then
    echo "Host $(hostname) is killing its arrow_helper process with PID: "$PID
    kill $PID
else
    echo "Host $(hostname) can not find a running arrow_helper process to kill."
fi

exit
