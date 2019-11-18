#!/bin/bash
###############################################################################################################
#                                                                                                             #
#                                  Cassandra Node Snapshot Recovery for Slurm                                 #
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

export C4S_HOME=$HOME/.c4s
CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg
CASS_HOME=$(cat $CFG_FILE | grep "CASS_HOME=" | sed 's/CASS_HOME=//g' | sed 's/"//g')
SNAP_ORIG=$(cat $CFG_FILE | grep "SNAP_PATH=" | sed 's/SNAP_PATH=//g' | sed 's/"//g')
ROOT_PATH=${1}
UNIQ_ID=${2}
DATA_HOME=$ROOT_PATH/c4j/cassandra-data
CASSFILE=$C4S_HOME/casslist-"$UNIQ_ID".txt
RECOVER_FILE=$C4S_HOME/cassandra-recover-file-"$UNIQ_ID".txt

#get & set the token list (safely)
RECOVERY=$(cat $RECOVER_FILE)
if [ "$RECOVERY" != "" ]
then
    TKNFILE_LIST=$(find $SNAP_ORIG -type f -name $RECOVERY-ring.txt)
    echo "TKNFILE_LIST: "$TKNFILE_LIST
    OLD_NODES=$(echo $TKNFILE_LIST | sed 's+/+ +g' | sed "s+$RECOVERY-ring.txt++g" | awk '{ print $NF }')
    filecounter=0
    oldcounter=0
    if [ "$(echo $OLD_NODES | grep $(hostname))" != "" ]
    then
        if [ -d $ROOT_PATH/../$RECOVERY ]; then
            # If the data is still in the node just moves it to the new data folder
            mv $ROOT_PATH/../$RECOVERY $ROOT_PATH
	    exit
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
                sed -i "s/.*initial_token:.*/initial_token: $(cat $tokens)/" $C4S_HOME/conf/cassandra-$(hostname).yaml

                orig_host=$(echo $tokens | sed "s+$SNAP_ORIG++g" | sed 's+/+ +g' | awk '{ print $1 }')
                clean_token=$RECOVERY
                for folder in $(find $SNAP_ORIG/$orig_host/$clean_token -maxdepth 1 -type d | sed -e "1d")
                do
                    clean_folder=$(echo $folder | sed 's+/+ +g' | awk '{ print $NF }') 
		    echo "CLEAN_FOLDER: "$clean_folder
                    mkdir $DATA_HOME/$clean_folder
                    for subfolder in $(find $SNAP_ORIG/$orig_host/$clean_token/$clean_folder -maxdepth 1 -type d | sed -e "1d")
                    do
                        clean_subfolder=$(echo $subfolder | sed 's+/+ +g' | awk '{ print $NF }')
                        mkdir $DATA_HOME/$clean_folder/$clean_subfolder
                        echo "COPY TO: cp $SNAP_ORIG/$orig_host/$clean_token/$clean_folder/$clean_subfolder/snapshots/$RECOVERY/* $DATA_HOME/$clean_folder/$clean_subfolder/"
                        cp $SNAP_ORIG/$orig_host/$clean_token/$clean_folder/$clean_subfolder/snapshots/$RECOVERY/* $DATA_HOME/$clean_folder/$clean_subfolder/
                    done
                done
                break
            fi
        fi  
    done
fi
