#!/bin/bash
###############################################################################################################
#                                                                                                             #
#                                  Cassandra Node Snapshot Launcher for Slurm                                 #
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

C4S_HOME=$HOME/.c4s
CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg
CASS_HOME=$(cat $CFG_FILE | grep "CASS_HOME=" | sed 's/CASS_HOME=//g' | sed 's/"//g')
SNAP_PATH=$(cat $CFG_FILE | grep "SNAP_PATH=" | sed 's/SNAP_PATH=//g' | sed 's/"//g')
SNAP_NAME=${1}
ROOT_PATH=${2}
CLUSTER=${3}
UNIQ_ID=${4}
DATA_HOME=$ROOT_PATH/c4j/cassandra-data
SNAP_DEST=$SNAP_PATH/$(hostname)/$SNAP_NAME
SNAP_STATUS_FILE=$C4S_HOME/snap-status-$SNAP_NAME-$(hostname)-file.txt
CASSFILE=$C4S_HOME/casslist-"$UNIQ_ID".txt
HST_IFACE="-ib0" #interface configured in the cassandra.yaml file

if [ $(cat $CASSFILE | grep $(hostname)) != "" ]; then
    # Flushing before taking the snapshot
    $CASS_HOME/bin/nodetool flush

    # It launches the snapshot
    $CASS_HOME/bin/nodetool snapshot -t $SNAP_NAME

    # If the main snapshots directory does not exist, it is created
    mkdir -p $SNAP_PATH

    # If the snapshot directory for this host does not exist, it is created
    mkdir -p $SNAP_PATH/$(hostname)

    # Creates the destination directory for this snapshot
    mkdir -p $SNAP_DEST

    # It also saves the tokens assigned to this host
    $CASS_HOME/bin/nodetool ring | grep $(cat /etc/hosts | grep $(hostname)"$HST_IFACE" | awk '{ print $1 }') | awk '{print $NF ","}' | xargs > $SNAP_DEST/../$SNAP_NAME-ring.txt

    # Saving cluster name (to restore it properly)
    echo "$CLUSTER" > $SNAP_DEST/../$SNAP_NAME-cluster.txt

    # For each folder in the data home, it is checked if the links to GPFS are already created, otherwise, create it
    for folder in $(ls $DATA_HOME)
    do
        mkdir -p $SNAP_DEST/$folder
        for subfolder in $(ls $DATA_HOME/$folder)
        do
            mkdir -p $SNAP_DEST/$folder/$subfolder
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
