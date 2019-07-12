#!/bin/bash
###############################################################################################################
#                                                                                                             #
#                                  Cassandra Node Snapshot Launcher for Slurm                                 #
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

C4S_HOME=$HOME/.c4s
CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg
CASS_HOME=$(cat $CFG_FILE | grep -v "#" | grep "CASS_HOME=" | tail -n 1 | sed 's/CASS_HOME=//g' | sed 's/"//g' | sed "s/'//g")
SNAP_PATH=$(cat $CFG_FILE | grep -v "#" | grep "SNAP_PATH=" | tail -n 1 | sed 's/SNAP_PATH=//g' | sed 's/"//g' | sed "s/'//g")
SNAP_NAME=${1}
ROOT_PATH=${2}
CLUSTER=${3}
UNIQ_ID=${4}
DATA_HOME=$ROOT_PATH/cassandra-data
SNAP_DEST=$SNAP_PATH/$SNAP_NAME/$(hostname)
SNAP_STATUS_FILE=$C4S_HOME/snap-status-$SNAP_NAME-$(hostname)-file.txt
CASSFILE=$C4S_HOME/casslist-"$UNIQ_ID".txt
RINGFILE=$C4S_HOME/ringfile-"$UNIQ_ID".txt
RINGDONE=$C4S_HOME/ringdone-"$UNIQ_ID".txt
HST_IFACE="-ib0" #interface configured in the cassandra.yaml file

if [ $(cat $CASSFILE | grep $(hostname)) != "" ]; then
    # Flushing before taking the snapshot
    $CASS_HOME/bin/nodetool flush

    # It launches the snapshot
    $CASS_HOME/bin/nodetool snapshot -t $SNAP_NAME

    # If the main snapshots directory does not exist, it is created
    mkdir -p $SNAP_PATH

    # Creates the destination directory for this snapshot
    mkdir -p $SNAP_DEST

    # If this is the first node it gets the full ring input
    if [ "$(head -n 1 $CASSFILE | grep $(hostname))" != "" ]; then
        rm -f $RINGFILE $RINGDONE
        $CASS_HOME/bin/nodetool ring > $RINGFILE
        echo "1" > $RINGDONE
    fi

    # It also saves the tokens assigned to this host, it waits until the ring file is ready
    while [ ! -s $RINGDONE ]; do
        sleep 1
    done
    cat $RINGFILE | grep -F " $(cat /etc/hosts | grep $(hostname)"$HST_IFACE" | awk '{ print $1 }') " | awk '{print $NF ","}' | xargs > $SNAP_DEST/$SNAP_NAME-ring.txt

    # Saving cluster name (to restore it properly)
    echo "$CLUSTER" > $SNAP_DEST/$SNAP_NAME-cluster.txt

    # For each folder in the data home, it is checked if the links to GPFS are already created, otherwise, create it
    for folder in $(ls $DATA_HOME)
    do
        for subfolder in $(ls $DATA_HOME/$folder)
        do
            mkdir -p $SNAP_DEST/$folder/$subfolder/snapshots
            if [ -d $DATA_HOME/$folder/$subfolder/snapshots/$SNAP_NAME ]
            then
                mv $DATA_HOME/$folder/$subfolder/snapshots/$SNAP_NAME $SNAP_DEST/$folder/$subfolder/snapshots/
                ln -s $SNAP_DEST/$folder/$subfolder/snapshots/$SNAP_NAME $DATA_HOME/$folder/$subfolder/snapshots/$SNAP_NAME
            fi
        done
    done
    # When it finishes creates the DONE status file for this host
    echo "DONE" > $SNAP_STATUS_FILE
else
    echo "${0}: Node $(hostname) is not part of the Cassandra cluster. Skipping..."
fi
