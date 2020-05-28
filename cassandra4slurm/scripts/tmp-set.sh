#!/bin/bash
CASS_HOME=${1}
DATA_PATH=${2}
COMM_PATH=${3}
SAV_CACHE=${4}
ROOT_PATH=${5}
CLUSTER=${6}
UNIQ_ID=${7}
export C4S_HOME=$HOME/.c4s
#CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg
#export CASS_HOME=$(cat $CFG_FILE | grep -v "#" | grep "CASS_HOME=" | tail -n 1 | sed 's/CASS_HOME=//g' | sed 's/"//g' | sed "s/'//g")
CASSFILE=$C4S_HOME/casslist-"$UNIQ_ID".txt
TEMPLATE_FILE=$C4S_HOME/conf/template-aux-"$SLURM_JOB_ID".yaml
NODE_YAML=$C4S_HOME/conf/cassandra-$(hostname).yaml

# Building directory tree
mkdir -p $ROOT_PATH/

# If the data path exists, cleans the content, otherwise it is created
# It gives group write permissions by default 
if [ -d $DATA_PATH ]; then
    rm -rf $DATA_PATH/*
fi
mkdir -p $DATA_PATH/hints

# Commit Log folder reset
# It gives group write permissions by default
# By default it is /tmp/cassandra-commitlog, if you change it you should also change the cassandra.yaml file
if [ -d $COMM_PATH ]; then
    rm -rf $COMM_PATH/*
fi
mkdir -p $COMM_PATH

# Set the cluster name and data path in the config file (safely)
sed "s/.*cluster_name:.*/cluster_name: \'$CLUSTER\'/g" $TEMPLATE_FILE | sed "s+.*hints_directory:.*+hints_directory: $DATA_PATH/hints+" | sed 's/.*data_file_directories.*/data_file_directories:/' | sed "/data_file_directories:/!b;n;c     - $DATA_PATH" | sed "s+.*commitlog_directory:.*+commitlog_directory: $COMM_PATH+" | sed "s+.*saved_caches_directory:.*+saved_caches_directory: $SAV_CACHE+g" > $NODE_YAML

if [ "$(hostname)" == "$(head -n 1 $CASSFILE)" ] || [ "$(hostname)" == "$(head -n 2 $CASSFILE | tail -n 1)" ]; then
    sed -i 's/auto_bootstrap:.*//g' $NODE_YAML
    echo >> $NODE_YAML
    echo "auto_bootstrap: false" >> $NODE_YAML
fi
