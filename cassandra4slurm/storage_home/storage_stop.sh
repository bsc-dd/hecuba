#!/bin/bash
###############################################################################################################
#                                                                                                             #
#                                             Cassandra Job for HPC                                           #
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


# Parameters
UNIQ_ID=${1}          # Unique ID to identify related files
C4S_HOME=$HOME/.c4s
MODULE_PATH=$HECUBA_ROOT/bin/cassandra4slurm


#CASSANDRA_NODES=${2}  # Number of Cassandra nodes to spawn
# I guess we dont need any of these.
SNAPSHOT_FILE=$C4S_HOME/cassandra-snapshot-file-"$UNIQ_ID".txt # EXPORTS IN STORAGE_INIT ARE NOT AVAILABLE HERE? LOL
source $MODULE_PATH/hecuba_debug.sh
DBG "  Checking if taking a snapshot is necessary..."
DBG "  SNAPSHOT_FILE=$SNAPSHOT_FILE"

# If an snapshot was ordered, it is done
if [[ -f $SNAPSHOT_FILE   &&  "$(cat $SNAPSHOT_FILE)" != "" ]]
then
    source $SNAPSHOT_FILE 
    TIME1=`date +"%T.%3N"`
    SNAP_NAME="$THETIME"

    source $MODULE_PATH/snapshot.sh $SNAP_NAME $ROOT_PATH $CLUSTER $UNIQ_ID

    SNAP_CONT=0
    while [ "$SNAP_CONT" != "$N_NODES" ]
    do
        SNAP_CONT=0
        for u_host in $casslist
        do
            if [ -f $C4S_HOME/snap-status-$SNAP_NAME-$u_host-file.txt ]
            then
                SNAP_CONT=$(($SNAP_CONT+1))
            fi
        done
    done

    TIME2=`date +"%T.%3N"`

    DBG "[STATS] Snapshot initial datetime: $TIME1"
    DBG "[STATS] Snapshot final datetime: $TIME2"

    show_time "[STATS] Snapshot process took: " $TIME1 $TIME2

    # Cleaning status files
    rm -f $C4S_HOME/snap-status-$SNAP_NAME-*-file.txt
fi
FINISHED=$C4S_HOME/stop."$UNIQ_ID".txt
if [[ -f $FINISHED && "$(cat $FINISHED)" == "1" ]]; then
    DBG " CASSANDRA_NODELIST=$CASSANDRA_NODELIST"
    DBG " N_NODES           =$N_NODES"
    #if cassandra was launched with the application then it exists a file stop with a 1 inside to kill it
    srun  --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 bash $MODULE_PATH/killer.sh
fi
