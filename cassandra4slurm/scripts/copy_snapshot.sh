#!/bin/bash

SNAP_NAME=${1}
ROOT_PATH=${2}
CLUSTER=${3}
UNIQ_ID=${4}

C4S_HOME=$HOME/.c4s

DATA_HOME=$ROOT_PATH/cassandra-data
CASSFILE=$C4S_HOME/casslist-"$UNIQ_ID".txt
RINGFILE=$C4S_HOME/ringfile-"$UNIQ_ID".txt
RINGDONE=$C4S_HOME/ringdone-"$UNIQ_ID".txt
HST_IFACE="-ib0" #interface configured in the cassandra.yaml file


source $HECUBA_ROOT/bin/cassandra4slurm/hecuba_debug.sh

DBG " Current SNAP_PATH [$SNAP_PATH]"
SNAP_DEST=$SNAP_PATH/$SNAP_NAME/$(hostname)
SNAP_STATUS_FILE=$C4S_HOME/snap-status-$SNAP_NAME-$(hostname)-file.txt

    # Creates the destination directory for this snapshot
mkdir -p $SNAP_DEST
while [ ! -s $RINGDONE ]; do
    echo " [INFO] Current RINGDONE [$RINGDONE] non existent"
    sleep 1
done
DBG "  Current RINGFILE $RINGFILE -> $(hostname) $SNAP_DEST/$SNAP_NAME-ring.txt"
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

# If HECUBA_ARROW is enabled, copy the ARROW directory
if [ ! -z $HECUBA_ARROW ]; then
    DBG " HECUBA ARROW is enabled"
    DBG "    HECUBA_ARROW_PATH $HECUBA_ARROW_PATH/arrow"
    DBG "    -> SNAP_DEST      $SNAP_DEST/.arrow"
    #cp -Rf $HECUBA_ARROW_PATH/arrow $SNAP_DEST/.arrow
    # In GPFS small files are the worst, therefore creata single file with everything
    pushd $HECUBA_ARROW_PATH/arrow
    tar czf $SNAP_DEST/.arrow.tar.gz .
    popd
    DBG " Snapshot $SNAP_DEST/.arrow.tar.gz generated"
    echo "# HECUBA_ARROW Enabled" >> $SNAP_DEST/hecuba_environment.txt
fi

# When it finishes creates the DONE status file for this host
echo "DONE" > $SNAP_STATUS_FILE
