#!/bin/bash
###############################################################################################################
#                                                                                                             #
#                                   Cassandra Node Snapshot Recovery for HPC                                  #
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

export C4S_HOME=$HOME/.c4s
CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg
source $CFG_FILE
SNAP_ORIG=$SNAP_PATH
ROOT_PATH=${1}
UNIQ_ID=${2}
DATA_HOME=$ROOT_PATH/cassandra-data
CASSFILE=$C4S_HOME/casslist-"$UNIQ_ID".txt
RECOVER_FILE=$C4S_HOME/cassandra-recover-file-"$UNIQ_ID".txt

source $HECUBA_ROOT/bin/cassandra4slurm/hecuba_debug.sh

# Get & set the token list (safely)
RECOVERY=$(cat $RECOVER_FILE)
if [ "$RECOVERY" != "" ]
then
    TKNFILE_LIST=$(find $SNAP_ORIG/$RECOVERY -type f -name $RECOVERY-ring.txt)
    DBG "TKNFILE_LIST: "$TKNFILE_LIST

    OLD_NODES=$(echo $TKNFILE_LIST | sed 's+/+ +g' | sed "s+$RECOVERY-ring.txt++g" | awk '{ print $NF }')
    filecounter=0
    oldcounter=0
    if [ "$(echo $OLD_NODES | grep $(hostname))" != "" ]
    then
        if [ -d "$ROOT_PATH/../$RECOVERY/" ]; then
            # If the data is still in the node just moves it to the new data folder
            mv $ROOT_PATH/../$RECOVERY/ $ROOT_PATH/
	    unset TKNFILE_LIST
        else
            # If the data is not there anymore reduces the token list to the snapshot of this node
            TKNFILE_LIST=$(echo $TKNFILE_LIST | grep $(hostname))
        fi 
    fi

    for tokens in $TKNFILE_LIST
    do  
        ((filecounter++))
        nodeidx=$((filecounter - oldcounter))
        if [ "$(cat $CASSFILE | sed -n ""$nodeidx"p")" == "$(hostname)" ] || [ "$(echo $TKNFILE_LIST | wc -w)" == "1" ]
        then
            old_node=$(echo $tokens | sed 's+/+ +g' | sed "s+$RECOVERY-ring.txt++g" | awk '{ print $NF }')
            if [ "$(cat $CASSFILE | grep $old_node)" != "" ] && [ "$old_node" != "$(hostname)" ]; then
                ((oldcounter++))
            else
                DBG "Restoring snapshot in node #"$filecounter": "$(hostname)
                ORIGINAL_CLUSTER_NAME=$(cat $SNAP_ORIG/$RECOVERY/$old_node/$RECOVERY-cluster.txt)
                sed -i "s/.*cluster_name:.*/cluster_name: \'$ORIGINAL_CLUSTER_NAME\'/g" $C4S_HOME/conf/cassandra-$(hostname).yaml
                sed -i "s/.*initial_token:.*/initial_token: $(cat $tokens)/" $C4S_HOME/conf/cassandra-$(hostname).yaml

                for folder in $(find $SNAP_ORIG/$RECOVERY/$old_node -maxdepth 1 -type d | sed -e "1d")
                do
                    clean_folder=$(echo $folder | sed 's+/+ +g' | awk '{ print $NF }') 
                    DBG "CLEAN_FOLDER: "$clean_folder
                    mkdir $DATA_HOME/$clean_folder
                    for subfolder in $(find $SNAP_ORIG/$RECOVERY/$old_node/$clean_folder -maxdepth 1 -type d | sed -e "1d")
                    do
                        clean_subfolder=$(echo $subfolder | sed 's+/+ +g' | awk '{ print $NF }')
                        mkdir $DATA_HOME/$clean_folder/$clean_subfolder
                        DBG "COPY TO: cp $SNAP_ORIG/$RECOVERY/$old_node/$clean_folder/$clean_subfolder/snapshots/$RECOVERY/* $DATA_HOME/$clean_folder/$clean_subfolder/"
                        cp $SNAP_ORIG/$RECOVERY/$old_node/$clean_folder/$clean_subfolder/snapshots/$RECOVERY/* $DATA_HOME/$clean_folder/$clean_subfolder/
                    done
                done
                # Recover Arrow files if needed
                if [ ! -z $HECUBA_ARROW ]; then
                    if [ ! -f $SNAP_ORIG/$RECOVERY/$old_node/hecuba_environment.txt ]; then
                        echo "HECUBA_ARROW is enabled, but snapshot is not ARROW enabled!"
                        echo " Exitting"
                        exit
                    fi

                    # Set HECUBA_ARROW_PATH
                    export HECUBA_ARROW_PATH=$ROOT_PATH/

                    echo "[INFO] HECUBA_ARROW is enabled"
                    echo "[INFO]    SNAP_DEST            $SNAP_ORIG/$RECOVERY/$old_node/.arrow"
                    echo "[INFO]    -> HECUBA_ARROW_PATH $HECUBA_ARROW_PATH/arrow"
                    mkdir -p $HECUBA_ARROW_PATH/arrow
                    tar zxf $SNAP_ORIG/$RECOVERY/$old_node/.arrow.tar.gz -C $HECUBA_ARROW_PATH/arrow
                    echo "[INFO] Recovered Arrow files in $(hostname)"
                fi
                break
            fi
        fi  
    done
fi
