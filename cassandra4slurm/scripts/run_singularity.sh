#!/bin/bash
###############################################################################################################
#                                                                                                             #
#                                             Cassandra Job for HPC                                           #
#                                    Yolanba Becerra - yolanba.becerra@bsc.es                                 #
#                                     Juanjo Costa - juan.jose.costa@upc.edu                                  #
#                                                                                                             #
#                                        Barcelona Supercomputing Center                                      #
#                                                    .-.--_                                                   #
#                                                  ,´,´.´   `.                                                #
#                                                  | | | BSC |                                                #
#                                                  `.`.`. _ .´                                                #
#                                                    `·`··                                                    #
#                                                                                                             #
###############################################################################################################

if [ "x$HECUBA_ROOT" == "x" ]; then
    echo "[ERROR] HECUBA_ROOT not defined. Is Hecuba module loaded?"
    exit
fi

source $HECUBA_ROOT/bin/cassandra4slurm/hecuba_debug.sh

DBG " RECEIVED ARGS ARE:"
DBG " $@"
# Parameters
UNIQ_ID=${1}            # Unique ID to identify related files (in this case the JOBID)
CASSANDRA_CONF=${2}     # Directory with the cassandra configuration DIRECTORY
XCASSPATH=${3}          # Cassandra installation directory (inside container)
LOG_DIR=${4}            # Directory to store cassandra logs
SINGULARITYIMG=${5}     # Singularity image to run
IFACE=${6}              # Network interface to use

export C4S_HOME=$HOME/.c4s

DBG " UNIQ_ID       =[$UNIQ_ID]"
DBG " CASSANDRA_CONF=[$CASSANDRA_CONF]"
DBG " XCASSPATH     =[$XCASSPATH]"
DBG " LOG_DIR       =[$LOG_DIR]"
DBG " SINGULARITYIMG=[$SINGULARITYIMG]"
DBG " IFACE         =[$IFACE]"

# Calculate hostname IP to know the file to access
HOSTNAMEIP=$(hostname)
HOSTNAMEIP=$(get_node_ip $HOSTNAMEIP $IFACE)
singularity run  \
            --env CASSANDRA_CONF=${CASSANDRA_CONF} \
            --env LOCAL_JMX='no' \
            -B ${LOG_DIR}:${XCASSPATH}/logs \
            ${SINGULARITYIMG} \
            ${XCASSPATH}/bin/cassandra -f \
                -Dcassandra.consistent.rangemovement=false \
                -Dcassandra.config=file://$CASSANDRA_CONF/cassandra-$HOSTNAMEIP.yaml -f | awk "{ print  \"$HOSTNAMEIP\",\$0 }"
