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

# Get & set the token list (safely)
RECOVERY=$(cat $RECOVER_FILE)
if [ "$RECOVERY" != "" ]
then
    TKNFILE_LIST=$(find $SNAP_ORIG/$RECOVERY -type f -name $RECOVERY-ring.txt)
    echo "TKNFILE_LIST: "$TKNFILE_LIST
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
                echo "Restoring snapshot in node #"$filecounter": "$(hostname)
                ORIGINAL_CLUSTER_NAME=$(cat $SNAP_ORIG/$RECOVERY/$old_node/$RECOVERY-cluster.txt)
                sed -i "s/.*cluster_name:.*/cluster_name: \'$ORIGINAL_CLUSTER_NAME\'/g" $C4S_HOME/conf/cassandra-$(hostname).yaml
                sed -i "s/.*initial_token:.*/initial_token: $(cat $tokens)/" $C4S_HOME/conf/cassandra-$(hostname).yaml

                for folder in $(find $SNAP_ORIG/$RECOVERY/$old_node -maxdepth 1 -type d | sed -e "1d")
                do
                    clean_folder=$(echo $folder | sed 's+/+ +g' | awk '{ print $NF }') 
		    echo "CLEAN_FOLDER: "$clean_folder
                    mkdir $DATA_HOME/$clean_folder
                    for subfolder in $(find $SNAP_ORIG/$RECOVERY/$old_node/$clean_folder -maxdepth 1 -type d | sed -e "1d")
                    do
                        clean_subfolder=$(echo $subfolder | sed 's+/+ +g' | awk '{ print $NF }')
                        mkdir $DATA_HOME/$clean_folder/$clean_subfolder
                        echo "COPY TO: cp $SNAP_ORIG/$RECOVERY/$old_node/$clean_folder/$clean_subfolder/snapshots/$RECOVERY/* $DATA_HOME/$clean_folder/$clean_subfolder/"
                        cp $SNAP_ORIG/$RECOVERY/$old_node/$clean_folder/$clean_subfolder/snapshots/$RECOVERY/* $DATA_HOME/$clean_folder/$clean_subfolder/
                    done
                done
                break
            fi
        fi  
    done
fi
