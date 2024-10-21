#!/bin/bash
#pre-requisite: variable HECUBA_ARROW is set
UNIQ_ID=$1
LOGFILE=$2
export C4S_HOME=$HOME/.c4s
HECUBA_ENVIRON=$C4S_HOME/environ-$UNIQ_ID.txt

MODULE_PATH=$HECUBA_ROOT/bin/cassandra4slurm

source $MODULE_PATH/hecuba_debug.sh
source $HECUBA_ENVIRON

ARROW_HELPER=$HECUBA_ROOT/bin/arrow_helper
DBG " Launching Arrow helper at [$hostname]"
$ARROW_HELPER $LOGFILE &

