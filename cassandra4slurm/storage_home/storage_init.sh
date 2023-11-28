#!/bin/bash
###############################################################################################################
#                                                                                                             #
#                                             Cassandra Job for HPC                                           #
#                                          Eloy Gil - eloy.gil@bsc.es                                         #
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
export UNIQ_ID=${1}          # Unique ID to identify related files (in this case the JOBID)
MASTER_NODE=${2}      # Master node name (same as 3rd parameter)
WORKER_NODES=${4}     # Worker nodes list (separated by spaces)
NETWORK=${5}            # 'infiniband' or other(ethernet)
STORAGE_PROPS=${6}
FILE_TO_SET_ENV_VARS=${7}
SINGULARITY=${8}      # possible values:
                      #   "false": regular execution
                      #   "default": execution with the default singularity image
                      #    a path to a singularity image
CORES_MASK=${9}       # mask of cores to be used by cassandra: NOT YET IMPLEMENTED



DBG " UNIQ_ID IS        : $UNIQ_ID"
DBG " MASTER NODE IS    : $MASTER_NODE"
DBG " WORKER NODES ARE  : $WORKER_NODES"
DBG " STORAGE PROPS FILE: $STORAGE_PROPS"
DBG " FILE TO SET VARS IS ${FILE_TO_SET_ENV_VARS}"
DBG " SINGULARITY       : ${SINGULARITY}"
DBG " CORES_MASK        : ${CORES_MASK}"


if [ "x$WORKER_NODES" == "x" ]; then
        echo "[WARNING] Empty worker node lists, launching cassandra on the master node"
	WORKER_NODES=$MASTER_NODE
fi


SINGULARITYIMG="disabled"
CASSANDRA_NODES=$(echo $WORKER_NODES | wc -w)  # Number of Cassandra nodes to spawn (in this case in every node)
DBG " CASSANDRA NODES   : $CASSANDRA_NODES"
DISJOINT=0            # Guarantee disjoint allocation. 1: Yes, 0 or empty otherwise
DBG " DISJOINT          = $DISJOINT"
if [ "$DISJOINT" == "1" ]; then
    C4S_CASSANDRA_CORES=48
else
    C4S_CASSANDRA_CORES=4
fi

if [ "$(echo "$NETWORK" | sed 's/-//g')" == "infiniband" ]; then
    iface="ib0"
else
    iface="eth0"
fi
if [ "${SINGULARITY}" != "false" ]; then
    if [ "${SINGULARITY}" == "default" ]; then
        SINGULARITYIMG="$HECUBA_ROOT/singularity/cassandra"
    else
        SINGULARITYIMG=${SINGULARITY}
    fi
fi


RETRY_MAX=500 # This value is around 50, higher now to test big clusters that need more time to be completely discovered

MAKE_SNAPSHOT=0


DBG " STORAGE PROPS CONTENT:"
DBG $(cat $STORAGE_PROPS)
source $STORAGE_PROPS

export C4S_HOME=$HOME/.c4s
export HECUBA_ENVIRON=$C4S_HOME/conf/hecuba_environment
export CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg
export MODULE_PATH=$HECUBA_ROOT/bin/cassandra4slurm

SNAPSHOT_FILE=$C4S_HOME/cassandra-snapshot-file-"$UNIQ_ID".txt
RECOVER_FILE=$C4S_HOME/cassandra-recover-file-"$UNIQ_ID".txt
HECUBA_TEMPLATE_FILE=$MODULE_PATH/hecuba_environment.template
CASSFILE=$C4S_HOME/casslist-"$UNIQ_ID".txt

CASSANDRA_NODELIST=$(echo $WORKER_NODES |sed s/\ /,/g)
echo $CASSANDRA_NODELIST |tr "," "\n" > $CASSFILE
NODETOOL_HOST=$( get_first_node $CASSANDRA_NODELIST )-$iface
export N_NODES=$CASSANDRA_NODES

export THETIME=$(date "+%Y%m%dD%H%Mh%Ss")"-$SLURM_JOB_ID"


[ ! -d ${C4S_HOME}/${UNIQ_ID} ] && mkdir -p ${C4S_HOME}/${UNIQ_ID}

function set_workspace () {
    mkdir -p $C4S_HOME/logs
    mkdir -p $C4S_HOME/conf
    DEFAULT_DATA_PATH=/scratch/tmp
    echo "#This is a Cassandra4Slurm configuration file. Every variable must be set and use an absolute path." > $CFG_FILE
    echo "# LOG_PATH is the default log directory." >> $CFG_FILE
    echo "LOG_PATH=\"$HOME/.c4s/logs\"" >> $CFG_FILE
    echo "# DATA_PATH is a path to be used to store the data in every node. Using the SSD local storage of each node is recommended." >> $CFG_FILE
    echo "DATA_PATH=\"$DEFAULT_DATA_PATH\"" >> $CFG_FILE
    echo "CASS_HOME=\"\$HECUBA_ROOT/cassandra-d8tree\"" >> $CFG_FILE
    echo "# SNAP_PATH is the destination path for snapshots." >> $CFG_FILE
    #echo "SNAP_PATH=\"/gpfs/projects/$(groups | awk '{ print $1 }')/$(whoami)/snapshots\"" >> $CFG_FILE
    echo "SNAP_PATH=\"$DEFAULT_DATA_PATH/hecuba/snapshots\"" >> $CFG_FILE
}

if [ ! -f $CFG_FILE ]; then
    set_workspace
    echo "INFO: A default Cassandra4Slurm config has been generated. Adapt the following file if needed and try again:"
    echo "$CFG_FILE"
    exit
fi

source $CFG_FILE
if [ "0$LOGS_DIR" == "0" ]; then
	#yolandab
    DEFAULT_LOGS_DIR=$(cat $CFG_FILE | grep "LOG_PATH=")
    if [ $? -eq 1 ]; then
            DEFAULT_LOGS_DIR=$PWD
    else
            DEFAULT_LOGS_DIR=$(echo $DEFAULT_LOGS_DIR| sed 's/LOG_PATH=//g' | sed 's/"//g')
    fi
    echo "[INFO] This execution will use $DEFAULT_LOGS_DIR as logging dir"
    #was:
    #DEFAULT_LOGS_DIR=$(cat $CFG_FILE | grep "LOG_PATH=" | sed 's/LOG_PATH=//g' | sed 's/"//g')
    LOGS_DIR=$DEFAULT_LOGS_DIR
fi
export CASS_HOME
export DATA_PATH
export SNAP_PATH

export ROOT_PATH=$DATA_PATH/$THETIME
export DATA_HOME=$ROOT_PATH/cassandra-data

export ENVFILE=$C4S_HOME/environ-"$UNIQ_ID".txt
if [ ! -f $HECUBA_ENVIRON ]; then
    echo "[INFO] Environment variables to load NOT found at $HECUBA_ENVIRON"
    echo "[INFO] Generating file with DEFAULT values."
    mkdir -p $C4S_HOME/conf
    cp $HECUBA_TEMPLATE_FILE $HECUBA_ENVIRON
else
    echo "[INFO] Environment variables to load found at $HECUBA_ENVIRON"
    DBG " Generated FILE WITH HECUBA ENVIRON at  ${ENVFILE}"
fi

source $HECUBA_ENVIRON

# Inherit ALL hecuba environment variables
cat $HECUBA_ENVIRON > $ENVFILE

if [ "X$HECUBA_ARROW" != "X" ]; then
    #Set HECUBA_ARROW_PATH to the DATA_PATH/TIME
    export HECUBA_ARROW_PATH=$ROOT_PATH/
    echo "export HECUBA_ARROW_PATH=$ROOT_PATH/" >> $ENVFILE
fi
cat ${STORAGE_PROPS} >> $ENVFILE

#FIX issue with COMPSS: create the directory that will hold the env file if it
#   does not exist (this should be ALREADY done by COMPSs but in the
#   meantime...)
mkdir -p $(dirname "${FILE_TO_SET_ENV_VARS}")

cat "$ENVFILE" > "${FILE_TO_SET_ENV_VARS}"

# Disable keyspace and table creation in hecuba module initialization as will be done in 'initialize_hecuba.sh'
echo "export CREATE_SCHEMA=False #Generated by $0" >> "${FILE_TO_SET_ENV_VARS}"

if [ "$DISJOINT" == "1" ]; then
    C4S_CASSANDRA_CORES=48
else
    C4S_CASSANDRA_CORES=4
fi
RETRY_MAX=500 # This value is around 50, higher now to test big clusters that need more time to be completely discovered




function exit_bad_node_status () {
    # Exit after getting a bad node status.
    echo "$NODE_COUNTER/$N_NODES nodes UP. Cassandra Cluster Status: ERROR"
    echo "It was expected to find $N_NODES nodes UP nodes, found "$NODE_COUNTER"."
    echo "Exiting..."
    exit
}
function get_nodes_up () {
    NODE_COUNTER=$($CASS_HOME/bin/nodetool -h $NODETOOL_HOST status | sed 1,5d | sed '$ d' | awk '{ print $1 }' | grep "UN" | wc -l)
}

function check_cassandra_table_at () {
    local what="$1" # Target to 'describe' to check different states in cassandra
    local first_node="$2"
    local res=1
    local nretries=1
    while [ "$res" != "0" ] ; do
        echo " * Checking Cassandra [$what] is available @$first_node... $nretries/$RETRY_MAX"
        $CASS_HOME/bin/cqlsh $first_node -e "describe $what;" > /dev/null
        res=$?
        ((nretries++))
        if [ $nretries -ge $RETRY_MAX ]; then
            echo "[ERROR]: Too many retries checking [$what]. Cassandra seems unavailable!! ($CASS_HOME/bin/cassandra)"
            echo "Exiting..."
            exit
        fi
    done
}
function check_cassandra_table () {
    local what="$1" # Target to 'describe' to check different states in cassandra
    local first_node=`head -n1 $CASSFILE`"-$iface"
    local last_node=`tail -n1 $CASSFILE`"-$iface"
    check_cassandra_table_at ${what} ${first_node}
    check_cassandra_table_at ${what} ${last_node}
}

function check_cassandra_is_available () {
    check_cassandra_table "keyspaces"
    echo " * Cassandra is available!"
}

function launch_arrow_helpers () {
    # Launch the 'arrow_helper' tool at each node in NODES, and leave their logs in LOGDIR
    NODES=$1
    LOGDIR=$2
    if [ ! -d $LOGDIR ]; then
        DBG " Creating directory to store Arrow helper logs at [$LOGDIR]:"
        mkdir -p $LOGDIR
    fi
    ARROW_HELPER=$HECUBA_ROOT/src/hecuba_repo/build/arrow_helper
    ARROW_HELPER=$HECUBA_ROOT/bin/arrow_helper


    for i in $(cat $NODES); do
        DBG " Launching Arrow helper at [$i] Log at [$LOGDIR/arrow_helper.$i.out]:"
        #ssh $i $ARROW_HELPER >& $LOGDIR/arrow_helper.$i.out &
        ssh $i $ARROW_HELPER  $LOGDIR/arrow_helper.$i.out &
    done
    # TODO Review this
    #echo "INFO: Launching Arrow helper at [$NODES] Log at [$LOGDIR/arrow_helper.$i.out]:"
    #srun --nodelist $NODES --ntasks-per-node=1 --cpus-per-task=4 $ARROW_HELPER
}

function run_cass_singularity() {
    local CASSANDRA_NODELIST=${1}   # List of nodes separated by ',' where cassandra must be started
    local iface=${2}                # Network interface to use: 'infiniband' / any other
    local C4S_CASSANDRA_CORES=${3}  # Number of cores to use by cassandra
    local LOG_DIR=${4}              # Directory where the cassandra logs will be stored (hopefully local to the node)
    local DATA_PATH=${5}            # Directory where the cassandra data will be stored (hopefully local to the node)
    local CASS_CONF=${6}            # Directory where the cassandra configuration files will be stored (may be shared to all nodes)
    local SINGULARITYIMG=${7}       # Singularity image to use

    [ -z "$SINGULARITYIMG" ] && die "Missing parameter: run_singularity CASSANDRA_NODELIST=${CASSANDRA_NODELIST} iface=${iface} C4S_CASSANDRA_CORES=${C4S_CASSANDRA_CORES} LOG_DIR=${LOG_DIR} DATA_PATH=${DATA_PATH} CASS_CONF=${CASS_CONF} SINGULARITYIMG=${SINGULARITYIMG}"

    DBG " SINGULARITY: Launching container at [${SINGULARITYIMG}] "
    # Prepare directory to store singularity configuration files
    mkdir -p ${CASS_CONF}/singularity
    DBG " SINGULARITY: CREATED [${CASS_CONF}/singularity] directory for cassandra configuration"

    XCASSPATH=$(singularity run ${SINGULARITYIMG} which cassandra)
    XCASSPATH=$(dirname ${XCASSPATH})/..
    DBG " SINGULARITY: CASSANDRA PATH [${XCASSPATH}] "
    singularity run ${SINGULARITYIMG} cp -LRf ${XCASSPATH}/conf ${CASS_CONF}/singularity  || die " SINGULARITY: copy failed"

    # Modify 'cassandra.yaml' from cassandra container to adapt to MN
    local SEED=$(get_first_node ${CASSANDRA_NODELIST})-${iface}

    cp ${CASS_CONF}/singularity/conf/cassandra.yaml ${CASS_CONF}/singularity/conf/cassandra.yaml.orig
    cat ${CASS_CONF}/singularity/conf/cassandra.yaml.orig \
        | sed "s/.*seeds:.*/          - seeds: \"$SEED\"/" \
        | sed "s/.*rpc_interface:.*/rpc_interface: $iface/" \
        | sed "s/.*listen_interface:.*/listen_interface: $iface/" \
        | sed "s/.*listen_address:.*/#listen_address: localhost/" \
        | sed "s/.*broadcast_address:.*/#broadcast_address: localhost/" \
        | sed "s/.*rpc_address:.*/#rpc_address: localhost/" \
        | sed "s/.*initial_token:.*/#initial_token:/" \
        > ${CASS_CONF}/singularity/conf/cassandra.yaml

    # Disable JMX authentication (nodetool)
    cp ${CASS_CONF}/singularity/conf/cassandra-env.sh ${CASS_CONF}/singularity/conf/cassandra-env.sh.orig
    cat ${CASS_CONF}/singularity/conf/cassandra-env.sh.orig \
        | sed "s/jmxremote.authenticate=true/jmxremote.authenticate=false/" \
        >${CASS_CONF}/singularity/conf/cassandra-env.sh

    # Prepare LOG_DIR
    mkdir -p ${LOG_DIR}

    #for each node launch a singularity instance
    DBG "CASSANDRA_SEEDS SET TO ${SEED}"
    echo "STARTING UP CASSANDRA (Singularity)..."

    run srun \
        --overlap --mem=0 \
        --output ${LOG_DIR}/cassandra.output \
        --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=${C4S_CASSANDRA_CORES} --nodes=$N_NODES \
        singularity run  \
            --env CASSANDRA_SEEDS=$SEED \
            --env CASSANDRA_CONF=${CASS_CONF}/singularity/conf \
            --env LOCAL_JMX='no' \
            -B ${LOG_DIR}:${XCASSPATH}/logs,${DATA_PATH}:${XCASSPATH}/data \
            ${SINGULARITYIMG} \
            ${XCASSPATH}/bin/cassandra -f &
}

TIME_START=`date +"%T.%3N"` # Time to start cassandra
if [ "${SINGULARITYIMG}" != "disabled" ]; then
    # TODO: LOGS_DIR uses *current* directory by default. Meaning that all instances would map that directory (which means that the content of that directory will be bullshit as all the nodes will overwrite the same data)
    run run_cass_singularity ${CASSANDRA_NODELIST} ${iface} ${C4S_CASSANDRA_CORES} ${LOGS_DIR}/${UNIQ_ID} ${DATA_PATH} ${C4S_HOME}/conf/${UNIQ_ID} ${SINGULARITYIMG}
else

    #check if cassandra is already running and then just configure hecuba environment
    # CONTACT_NAMES sould be set in STORAGE_PROPS file
    if [[ $( grep -v ^# $STORAGE_PROPS | grep -q CONTACT_NAMES ) -eq 1 ]]; then
            firstnode=$( get_first_node $CONTACT_NAMES )
            source $MODULE_PATH/initialize_hecuba.sh  $firstnode
            #echo "CONTACT_NAMES=$CONTACT_NAMES"
            #echo "export CONTACT_NAMES=$CONTACT_NAMES" >> ${FILE_TO_SET_ENV_VARS}
            DBG " FILE TO EXPORT VARS IS  ${FILE_TO_SET_ENV_VARS}"
            DBG $(cat ${FILE_TO_SET_ENV_VARS})
            exit
    fi

    # Cassandra is not already running... starting it
    mkdir -p $SNAP_PATH
    COMM_HOME=$ROOT_PATH/cassandra-commitlog
    SAV_CACHE=$ROOT_PATH/saved_caches

    #check if the user wants to start from a stored snapshot
    if [ "X$RECOVER" !=  "X" ]; then
        DBG "Trying to recover $SNAP_PATH/$RECOVER"
        if [ ! -e $SNAP_PATH/$RECOVER ]; then
            echo "[ERROR] Unable to find snapshot at $SNAP_PATH/$RECOVER !"
            echo " Exitting"
            exit
        fi
        echo $RECOVER > $RECOVER_FILE
        export CLUSTER=$RECOVER
        RECOVERING=$CLUSTER
        echo "[INFO] Recovering snapshot: $RECOVERING"
    else
        export CLUSTER=$THETIME
    fi



    NODEFILE_REMOVEME=$C4S_HOME/hostlist-"$UNIQ_ID".txt

    #export APPFILE=$C4S_HOME/applist-"$UNIQ_ID".txt
    APPPATHFILE=$C4S_HOME/app-"$UNIQ_ID".txt
    PYCOMPSS_FLAGS_FILE=$C4S_HOME/pycompss-flags-"$UNIQ_ID".txt
    PYCOMPSS_FILE=$C4S_HOME/pycompss-"$UNIQ_ID".sh
    scontrol show hostnames $SLURM_NODELIST > $NODEFILE_REMOVEME

    # Generating nodefiles
    #yolandab: this assumes that the n initial nodes are worker nodes. We have the variable worker nodes with the nodes separated by blanks, for the run command we need to use comma as the separator
    #head -n $CASSANDRA_NODES $NODEFILE > $CASSFILE


    if [ ! -f $CASS_HOME/bin/cassandra ]; then
        echo "[ERROR]: Cassandra executable is not placed where it was expected. ($CASS_HOME/bin/cassandra)"
        echo "Exiting..."
        exit
    fi

    if [ ! -f $C4S_HOME/stop."$UNIQ_ID".txt ]; then
        #we will end cassandra when the application ends
        #TODO: add a configuration to storage_props to keep it alive
        echo "1" > $C4S_HOME/stop."$UNIQ_ID".txt
    fi

    echo "STARTING UP CASSANDRA..."
    DBG " I am $(hostname)."
    if [ $N_NODES -gt 1 ]; then
        export REPLICA_FACTOR=2
    else
        export REPLICA_FACTOR=1
    fi

    # If template is not there, it is copied from cassandra config folder
    if [ ! -s $C4S_HOME/conf/template.yaml ]; then
        cp $CASS_HOME/conf/cassandra.yaml $C4S_HOME/conf/template.yaml || die "Error copying $C4S_HOME/conf/cassandra.yaml"
    fi

    seedlist=`head -n 2 $CASSFILE`
    seeds=`echo $seedlist | sed "s/ /-$iface,/g"`
    seeds=$seeds-$iface #using only infiniband atm, will change later
    sed "s/.*seeds:.*/          - seeds: \"$seeds\"/" $C4S_HOME/conf/template.yaml \
        | sed "s/.*rpc_interface:.*/rpc_interface: $iface/" \
        | sed "s/.*listen_interface:.*/listen_interface: $iface/" \
        | sed "s/.*listen_address:.*/#listen_address: localhost/" \
        | sed "s/.*rpc_address:.*/#rpc_address: localhost/" \
        | sed "s/.*initial_token:.*/#initial_token:/" \
        > $C4S_HOME/conf/template-aux-"$SLURM_JOB_ID".yaml


    DBG $(ls -la $C4S_HOME/conf/template-aux-"$SLURM_JOB_ID".yaml)


    echo "Launching Cassandra in the following hosts: $CASSANDRA_NODELIST"

    # If a snapshot is needed

    if [ "$MAKE_SNAPSHOT" == "1" ]; then
        cat $HECUBA_ENVIRON > $SNAPSHOT_FILE
        echo "export CASSANDRA_NODELIST=$CASSANDRA_NODELIST" >> $SNAPSHOT_FILE
        echo "export C4S_CASSANDRA_CORES=$C4S_CASSANDRA_CORES" >> $SNAPSHOT_FILE
        echo "export N_NODES=$N_NODES" >> $SNAPSHOT_FILE
        echo "export THETIME=$THETIME" >> $SNAPSHOT_FILE
        echo "export ROOT_PATH=$ROOT_PATH" >> $SNAPSHOT_FILE
        echo "export CLUSTER=$CLUSTER" >> $SNAPSHOT_FILE
        echo "export MODULE_PATH=$MODULE_PATH" >> $SNAPSHOT_FILE
    fi

    # Clearing data from previous executions and checking symlink coherence
    run srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=4 --nodes=$N_NODES \
            $MODULE_PATH/tmp-set.sh $CASS_HOME $DATA_HOME $COMM_HOME $SAV_CACHE $ROOT_PATH $CLUSTER $UNIQ_ID
    sleep 5

    # Recover snapshot if any

    if [ "X$RECOVERING" != "X" ]; then
        DBG "Launchig recover.sh"
        run srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=1 --nodes=$N_NODES \
                $MODULE_PATH/recover.sh $ROOT_PATH $UNIQ_ID
    fi

    # Launching Cassandra in every node
    run srun --overlap --mem=0 \
            --output ${C4S_HOME}/${UNIQ_ID}/cassandra.output \
            --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=${C4S_CASSANDRA_CORES} --nodes=$N_NODES \
            $MODULE_PATH/enqueue_cass_node.sh $UNIQ_ID &
    sleep 5

    # Cleaning config template
    rm -f $C4S_HOME/conf/template-aux-"$SLURM_JOB_ID".yaml

fi # !SINGULARITY

# Checking cluster status until all nodes are UP (or timeout)
echo "Checking..."
RETRY_COUNTER=0
get_nodes_up
while [ "X$NODE_COUNTER" != "X$N_NODES" ] && [ $RETRY_COUNTER -lt $RETRY_MAX ]; do
    echo "$NODE_COUNTER/$N_NODES nodes UP. Retry #$RETRY_COUNTER"
    echo "Checking..."
    sleep 5
    get_nodes_up
    ((RETRY_COUNTER++))
done
if [ "X$NODE_COUNTER" == "X$N_NODES" ]
then
    TIME_END=`date +"%T.%3N"`
    echo "$NODE_COUNTER/$N_NODES nodes UP. Cassandra Cluster started successfully."

    show_time "[STATS] Cluster launching process took: " $TIME_START $TIME_END
else
    echo "[ERROR]: Cassandra Cluster RUN timeout. Check STATUS."
    exit_bad_node_status
fi

# THIS IS THE APPLICATION CODE EXECUTING SOME TASKS USING CASSANDRA DATA, ETC
echo "CHECKING CASSANDRA STATUS: "
$CASS_HOME/bin/nodetool -h $NODETOOL_HOST status

check_cassandra_is_available

firstnode=$( get_first_node $CASSANDRA_NODELIST )-${iface}
source $MODULE_PATH/initialize_hecuba.sh  $firstnode

CNAMES=$(sed ':a;N;$!ba;s/\n/,/g' $CASSFILE)
CNAMES=$(echo $CNAMES | sed "s/,/-$iface,/g")-$iface

# WARNING! CONTACT_NAMES MUST be IP numbers!
first=1
CONTACT_NAMES=""
COMMA=""
for node in $(cat $CASSFILE); do
    if [ "$first" == "1" ]; then
        first=0
    else
        COMMA=","
    fi
    IP=$(host $node-$iface |cut -d' ' -f 4) #GET IP Address
    CONTACT_NAMES="${CONTACT_NAMES}${COMMA}${IP}"
done
export CONTACT_NAMES=$CONTACT_NAMES
echo "CONTACT_NAMES=$CONTACT_NAMES"
echo "export CONTACT_NAMES=$CONTACT_NAMES" >> ${FILE_TO_SET_ENV_VARS}

DBG " FILE TO EXPORT VARS IS  ${FILE_TO_SET_ENV_VARS}"
DBG $(cat ${FILE_TO_SET_ENV_VARS})

#srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=4 --nodes=$N_NODES "bash export CONTACT_NAMES=$CONTACT_NAMES" &
PYCOMPSS_STORAGE=$C4S_HOME/pycompss_storage_"$UNIQ_ID".txt
echo $CNAMES | tr , '\n' > $PYCOMPSS_STORAGE # Set list of nodes (with interface) in PyCOMPSs file

if [ "$SINGULARITYIMG" == "disabled" ]; then

    if [ "X$STREAMING" != "X" ]; then
        if [ ${STREAMING,,} == "true" ]; then
            source $HECUBA_ROOT/bin/cassandra4slurm/launch_kafka.sh
            launch_kafka $CASSANDRA_NODELIST $UNIQ_ID $N_NODES
        fi
    fi

    launch_arrow_helpers $CASSFILE  $LOGS_DIR/$UNIQ_ID
fi # SINGULARITYIMG
