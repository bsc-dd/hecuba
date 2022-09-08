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
PID=$(ps aux | grep java | grep -v grep| grep cassandra | awk '{ print $2 }')
if [ "$PID" != "" ]; then
    echo "Host $(hostname) is killing its cassandra process with PID: "$PID
    kill $PID
fi

# Kill Arrow Helpers
PID=$(ps aux |grep arrow_helper | grep -v grep| awk '{print $2}')
if [ "$PID" != "" ]; then
    echo "Host $(hostname) is killing its arrow_helper process with PID: "$PID
    kill $PID
fi

# Kill Kafka
PID=$(ps aux |grep java | grep -v grep| grep kafka | awk '{print $2}')
if [ "$PID" != "" ]; then
    echo "Host $(hostname) is killing its kafka process with PID: "$PID
    kill $PID
fi

PID=$(ps aux |grep java | grep -v grep| grep zookeeper | awk '{print $2}')
if [ "$PID" != "" ]; then
    echo "Host $(hostname) is killing its zookeeper process with PID: "$PID
    kill $PID
fi

exit
