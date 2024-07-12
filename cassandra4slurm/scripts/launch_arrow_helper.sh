#|/bin/bash

LOGFILE=$1
export C4S_HOME=$HOME/.c4s
HECUBA_ENVIRON=$C4S_HOME/conf/hecuba_environment

MODULE_PATH=$HECUBA_ROOT/bin/cassandra4slurm
CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg

source $MODULE_PATH/hecuba_debug.sh
source $CFG_FILE
source $HECUBA_ENVIRON

[ "X$HECUBA_ARROW" == "X" ] && return

ARROW_HELPER=$HECUBA_ROOT/bin/arrow_helper
DBG " Launching Arrow helper at [$hostname]"
$ARROW_HELPER $LOGFILE &

