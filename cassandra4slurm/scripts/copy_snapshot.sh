#!/bin/bash

SNAP_NAME=${1}
ROOT_PATH=${2}
CLUSTER=${3}
UNIQ_ID=${4}

DATA_HOME=$ROOT_PATH/cassandra-data
CASSFILE=$C4S_HOME/casslist-"$UNIQ_ID".txt
RINGFILE=$C4S_HOME/ringfile-"$UNIQ_ID".txt
RINGDONE=$C4S_HOME/ringdone-"$UNIQ_ID".txt
HST_IFACE="-ib0" #interface configured in the cassandra.yaml file


SNAP_DEST=$SNAP_PATH/$SNAP_NAME/$(hostname)
SNAP_STATUS_FILE=$C4S_HOME/snap-status-$SNAP_NAME-$(hostname)-file.txt

    # Creates the destination directory for this snapshot
mkdir -p $SNAP_DEST
while [ ! -s $RINGDONE ]; do
    sleep 1
done
echo " [INFO] Current RINGFILE $RINGFILE -> $(hostname) $SNAP_DEST/$SNAP_NAME-ring.txt"
NODE_IP=$(cat /etc/hosts | grep $(hostname)"$HST_IFACE" |awk '{print $1}')

cat $RINGFILE | grep -F $NODE_IP | awk '{print $NF }' | tr "\n" "," > $SNAP_DEST/$SNAP_NAME-ring.txt

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
