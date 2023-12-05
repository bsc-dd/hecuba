#!/bin/bash
DATA_PATH=${1}
COMM_PATH=${2}
SAV_CACHE=${3}
ROOT_PATH=${4}
export C4S_HOME=$HOME/.c4s
#CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg
#export CASS_HOME=$(cat $CFG_FILE | grep -v "#" | grep "CASS_HOME=" | tail -n 1 | sed 's/CASS_HOME=//g' | sed 's/"//g' | sed "s/'//g")

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

# Saved caches directory
# If path exists, cleans the content, otherwise it is created
# It gives group write permissions by default
if [ -d $SAV_CACHE ]; then
    rm -rf $SAV_CACHE/*
fi
mkdir -p $SAV_CACHE

