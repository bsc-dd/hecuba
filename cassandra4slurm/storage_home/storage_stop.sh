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
echo "[INFO] Checking if taking a snapshot is necessary..."
echo "[DEBUG] SNAPSHOT_FILE=$SNAPSHOT_FILE"
# If an snapshot was ordered, it is done
if [ "$(cat $SNAPSHOT_FILE)" != "" ]
then
    source $SNAPSHOT_FILE 
    TIME1=`date +"%T.%3N"`
    SNAP_NAME="$THETIME"
    # Looping over the assigned hosts until the snapshots are confirmed
    srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=4 --nodes=$N_NODES bash $MODULE_PATH/snapshot.sh $SNAP_NAME $ROOT_PATH $CLUSTER $UNIQ_ID

    while [ "$(ls 2> /dev/null ~/.c4s/snap-status-$SNAP_NAME-*-file.txt | wc -l)" != "$N_NODES" ]; do
        sleep 0.2
    done
    
    TIME2=`date +"%T.%3N"`
    echo "[STATS] Snapshot initial datetime: $TIME1"
    echo "[STATS] Snapshot final datetime: $TIME2" 
    MILL1=$(echo $TIME1 | cut -c 10-12)
    MILL2=$(echo $TIME2 | cut -c 10-12)
    TIMESEC1=$(date -d "$TIME1" +%s)
    TIMESEC2=$(date -d "$TIME2" +%s)
    TIMESEC=$(( TIMESEC2 - TIMESEC1 ))
    MILL=$(( MILL2 - MILL1 ))

    # Adjusting seconds if necessary
    if [ $MILL -lt 0 ]
    then
        MILL=$(( 1000 + MILL ))
        TIMESEC=$(( TIMESEC - 1 ))
    fi

    echo "[STATS] Snapshot process took: "$TIMESEC"s. "$MILL"ms."

    # Cleaning temporal files
    rm -f $C4S_HOME/snap-status-$SNAP_NAME-*-file.txt
    rm -f $SNAPSHOT_FILE
fi
if [ -f $C4S_HOME/stop."$UNIQ_ID".txt ]; then
        if [ "$(cat $C4S_HOME/stop."$UNIQ_ID".txt)" == "1" ]; then
		#if cassandra was launched with the application then it exists a file stop with a 1 inside to kill it
		srun  --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 bash $MODULE_PATH/killer.sh
fi
