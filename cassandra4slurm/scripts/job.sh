#!/bin/bash
###############################################################################################################
#                                                                                                             #
#                                             Cassandra Job for HPC                                           #
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

iface=$(echo "$CASS_IFACE" | sed 's/-//g')

# Parameters
UNIQ_ID=${1}          # Unique ID to identify related files
CASSANDRA_NODES=${2}  # Number of Cassandra nodes to spawn
APP_NODES=${3}        # Number of nodes to run the application
APP_NTASKS=${4}       # Number of total instances of the application to run in the cluster
PYCOMPSS_APP=${5}     # Application execution using PyCOMPSs. 0: No, 1: Yes
WITHDLB=${6}          # If set to 1 use dlb to manage allocation between cassandra and the application
DISJOINT=${7}         # Guarantee disjoint allocation. 1: Yes, empty otherwise

export C4S_HOME=$HOME/.c4s
HECUBA_ENVIRON=$C4S_HOME/conf/${UNIQ_ID}/hecuba_environment
MODULE_PATH=$HECUBA_ROOT/bin/cassandra4slurm
CFG_FILE=$C4S_HOME/conf/${UNIQ_ID}/cassandra4slurm.cfg
NODEFILE=$C4S_HOME/hostlist-"$UNIQ_ID".txt

# CASSFILETOSYNC is a special file that is consulted by other scripts,
# to ensure the cluster is UP.
# CASSFILE is used while Cassandra is starting up.
export CASSFILE=$C4S_HOME/casslist-"$UNIQ_ID".txt
export CASSFILETOSYNC=$C4S_HOME/casslistDONE-"$UNIQ_ID".txt

export APPFILE=$C4S_HOME/applist-"$UNIQ_ID".txt
APPPATHFILE=$C4S_HOME/app-"$UNIQ_ID".txt
PYCOMPSS_FLAGS_FILE=$C4S_HOME/pycompss-flags-"$UNIQ_ID".txt
PYCOMPSS_FILE=$C4S_HOME/pycompss-"$UNIQ_ID".sh
SNAPSHOT_FILE=$C4S_HOME/cassandra-snapshot-file-"$UNIQ_ID".txt
RECOVER_FILE=$C4S_HOME/cassandra-recover-file-"$UNIQ_ID".txt
rm -f $NODEFILE $CASSFILE $CASSFILETOSYNC $APPFILE

source $CFG_FILE
export CASS_HOME
export DATA_PATH
export SNAP_PATH
#export CASS_HOME=$(cat $CFG_FILE | grep -v "#" | grep "CASS_HOME=" | tail -n 1 | sed 's/CASS_HOME=//g' | sed 's/"//g' | sed "s/'//g")
#export DATA_PATH=$(cat $CFG_FILE | grep -v "#" | grep "DATA_PATH=" | tail -n 1 | sed 's/DATA_PATH=//g' | sed 's/"//g' | sed "s/'//g") 
#export SNAP_PATH=$(cat $CFG_FILE | grep -v "#" | grep "SNAP_PATH=" | tail -n 1 | sed 's/SNAP_PATH=//g' | sed 's/"//g' | sed "s/'//g")
mkdir -p $SNAP_PATH
THETIME=$(date "+%Y%m%dD%H%Mh%Ss")"-$SLURM_JOB_ID"
ROOT_PATH=$DATA_PATH/$THETIME
DATA_HOME=$ROOT_PATH/cassandra-data
COMM_HOME=$ROOT_PATH/cassandra-commitlog
SAV_CACHE=$ROOT_PATH/saved_caches
CLUSTER=$THETIME
export ENVFILE=$C4S_HOME/environ-"$UNIQ_ID".txt

source $HECUBA_ENVIRON

echo "[INFO] Generated FILE WITH HECUBA ENVIRON at  ${ENVFILE}"
cat $HECUBA_ENVIRON > $ENVFILE

source $MODULE_PATH/hecuba_debug.sh

function get_numphyscores() {
	# nproc --all --> gives physical AND virtual cores, but slurm only accepts 'physical'
	lscpu -p=Core |grep -v '#'| sort -u |wc -l
}

if is_HECUBA_ARROW_enabled ; then
    #Set HECUBA_ARROW_PATH to the DATA_PATH/TIME
    if [ "X$HECUBA_ARROW_PATH" == "X" ]; then
	export HECUBA_ARROW_PATH=$DATA_PATH/
	echo "[INFO] HECUBA_ARROW_PATH is NOT defined. Setting HECUBA_ARROW_PATH to [$HECUBA_ARROW_PATH]"
    fi
    # Just append the cluster name
    export HECUBA_ARROW_PATH=$HECUBA_ARROW_PATH/$CLUSTER
    echo "[INFO] HECUBA_ARROW_PATH modified to be [$HECUBA_ARROW_PATH]"
    #export HECUBA_ARROW_PATH=$ROOT_PATH/
    echo "export HECUBA_ARROW_PATH=$HECUBA_ARROW_PATH/" >> $ENVFILE
fi

if [ "X$C4S_CASSANDRA_CORES" == "X" ]; then
    if [ "$DISJOINT" == "1" ]; then
        C4S_CASSANDRA_CORES=$(get_numphyscores)
    else
        C4S_CASSANDRA_CORES=4
    fi
fi

RETRY_MAX=3000 # This value is around 50, higher now to test big clusters that need more time to be completely discovered

# hostnames2IP(file): Transform all host names into 'iface' IPs for 'file'
function hostnames2IP() {
    local FILE="$1"
    echo "* Transforming [$FILE] to iface [$iface]"
    rm -f ${FILE}.ips
    for node in $(cat ${FILE}); do
        nodeIP=$(get_node_ip $node $iface)
        [ "$nodeIP" != "" ] || die "Device [$iface] not found in node [$node]. Modify CASS_IFACE in the 'cassandra4slurm.cfg' file"
        echo "** Node [$node == $nodeIP]"
        echo $nodeIP >> ${FILE}.ips
    done
}

# gen_mask FIRST/LAST NUM_CORES: Generate a mask of NUM_CORES cores starting from the beginning or from the end
function gen_mask () {
	local BEGIN="$1"
	local NUM_CORES="$2"
	local CORES_TOT=$(get_numphyscores)

	# 10 cores
	# gen_mask FIRST 4
	# 1111 0000 00
	# 1	001	1		0000000001
	# 2	003	3		0000000011
	# 3	007	7		0000000111
	# (2^NUM_CORES)-1
	# gen_mask LAST 1
	# 1	200	512	1*2^9	1000000000
	# 2	300	768	3*2^8	1100000000
	# 3	380	896	7*2^7	1110000000
	# ((2^NUM_CORES)-1)*(2^(CORES_TOT-NUM_CORES))
	if [ "$BEGIN" == "FIRST" ]; then
		MASK=$(echo "obase=16;(2^${NUM_CORES})-1"|bc)
		#echo "Generated Mask using $NUM_CORES starting cores: = $MASK"
	else # LAST
		MASK=$(echo "obase=16;((2^${NUM_CORES})-1)*(2^(${CORES_TOT}-${NUM_CORES}))"|bc)
		#echo "Generated Mask using last $NUM_CORES cores: = $MASK"
	fi
	echo $MASK
}

# Generating nodefiles
# =====================
# Each host may have different interfaces, and therefore we need to decide
# which one (IFACE) to use. In a magical world, using the IP of that interface
# would work in all cases, ... but Slurm requires hostnames (sigh), so we keep
# both.
# There will be 2 files, one with hostnames and one with IP numbers:
#   - hostnames: casslist-$UNIQ_ID.txt (CASSFILE)
#   - IP number: casslist-$UNIQ_ID.txt.ips (CASSFILE.ips)
scontrol show hostnames $SLURM_NODELIST > $NODEFILE
tail -n $CASSANDRA_NODES $NODEFILE > $CASSFILE
head -n $APP_NODES $NODEFILE > $APPFILE

# Obtain IPs for iface (keep a copy of original files)
hostnames2IP $CASSFILE
hostnames2IP $APPFILE
# Generate LIST with hostnames separated by ',' required by Slurm
export APPNODELIST=$(cat $APPFILE | tr '\n' ',' | sed "s/,$//g")


DBG " APP_NODES             ="$APP_NODES
DBG "     CORES x PROCESS   ="$C4S_APP_CORES
DBG " CASSANDRA_NODES       ="$CASSANDRA_NODES
DBG "     CORES x CASS_NODE ="$C4S_CASSANDRA_CORES
DBG " DISJOINT              ="$DISJOINT

N_NODES=$(cat $CASSFILE | wc -l)

if [ "$(cat $RECOVER_FILE)" != "" ]; then
    CLUSTER=$(cat $RECOVER_FILE)
else
    CLUSTER=$THETIME
fi

function exit_killjob () {
    # Traditional harakiri
    scancel $SLURM_JOBID
}

function exit_bad_node_status () {
    # Exit after getting a bad node status. 
    echo "Cassandra Cluster Status: ERROR"
    echo "It was expected to find $N_NODES nodes UP nodes, found "$NODE_COUNTER"."
    echo "Exiting..."
    exit_killjob
}

function get_nodes_up () {
    first_node="$1"
    NODE_COUNTER=$($CASS_HOME/bin/nodetool -Dcom.sun.jndi.rmiURLParsing=legacy -h $first_node status | sed 1,5d | sed '$ d' | awk '{ print $1 }' | grep "UN" | wc -l)
}

function check_cassandra_is_available () {
    local first_node="$1"
    local res=1
    local nretries=1

    TMP=$(mktemp -t hecuba.tmp.XXXXXX)
    echo "describe keyspaces;" > $TMP
    trap "cleanup $TMP" SIGINT SIGQUIT SIGABRT
    while [ "$res" != "0" ] ; do
        #echo " * Checking Cassandra is available... $nretries/$RETRY_MAX"
        $CASS_HOME/bin/cqlsh $first_node -f $TMP > /dev/null
        res=$?
        ((nretries++))
        if [ $nretries -ge $RETRY_MAX ]; then
            echo "ERROR: Too many retries. Cassandra seems unavailable!! ($CASS_HOME/bin/cassandra)"
            echo "Exiting..."
            exit
        fi
    done
    echo " * Cassandra is available!"
    cleanup $TMP
}

if [ ! -f $CASS_HOME/bin/cassandra ]; then
    echo "ERROR: Cassandra executable is not placed where it was expected. ($CASS_HOME/bin/cassandra)"
    echo "Exiting..."
    exit
fi

if [ ! -f $SNAPSHOT_FILE ]; then
    echo "ERROR: The file that sets the snapshot/not snapshot option is not placed where it was expected ($SNAPSHOT_FILE)"
    echo "Exiting..."
    exit
fi

if [ ! -f $C4S_HOME/stop."$UNIQ_ID".txt ]; then
    echo "0" > $C4S_HOME/stop."$UNIQ_ID".txt
fi

if [ "$(cat $RECOVER_FILE)" != "" ]; then
    if [ ! -e $SNAP_PATH/$CLUSTER ]; then
        echo "[ERROR] Unable to find snapshot at $SNAP_PATH/$CLUSTER !"
        echo " Exitting"
        exit
    fi
    RECOVERING=$(cat $RECOVER_FILE)
    echo "[INFO] Recovering snapshot: $RECOVERING"
fi

echo "STARTING UP CASSANDRA..."
DBG " I am $(hostname)."
#export REPLICA_FACTOR=2

export CASS_CONF=$C4S_HOME/conf/${UNIQ_ID}

export CASS_YAML_FILE=${CASS_CONF}/cassandra.yaml
export CASS_ENV_FILE=${CASS_CONF}/cassandra-env.sh
export TEMPLATE_CASS_YAML_FILE=${CASS_CONF}/template.yaml.orig
export TEMPLATE_CASS_ENV_FILE=${CASS_CONF}/cassandra-env.sh.orig
cp $CASS_YAML_FILE $TEMPLATE_CASS_YAML_FILE
cp $CASS_ENV_FILE $TEMPLATE_CASS_ENV_FILE

casslist=`cat $CASSFILE`
seeds=`head -n 1 $CASSFILE.ips`

# GENERATE CASSANDRA CONFIGURATION FILES
# Do NOT use hostname! There are machines that return a differnt name than the one used in SLURM.
#only one node needs to do this, all the nodes share the same file
cat $TEMPLATE_CASS_YAML_FILE \
    | sed "s/.*cluster_name:.*/cluster_name: \'$CLUSTER\'/g" \
    | sed "s/.*num_tokens:.*/num_tokens: 256/g" \
    | sed "s+.*hints_directory:.*+hints_directory: $DATA_HOME/hints+" \
    | sed 's/.*data_file_directories.*/data_file_directories:/' \
    | sed "/data_file_directories:/!b;n;c     - $DATA_HOME" \
    | sed "s+.*commitlog_directory:.*+commitlog_directory: $COMM_HOME+" \
    | sed "s+.*saved_caches_directory:.*+saved_caches_directory: $SAV_CACHE+g" \
    | sed "s/.*seeds:.*/          - seeds: \"$seeds\"/" \
    | sed "s/.*broadcast_address:.*/#broadcast_address: localhost/" \
    | sed "s/.*initial_token:.*/#initial_token:/" \
    > ${CASS_YAML_FILE}
    echo "auto_bootstrap: false" >> ${CASS_YAML_FILE}

# Generate a configuration file for each cassandra node (used in recover)
for i in $(cat $CASSFILE.ips); do
    cat ${CASS_YAML_FILE} \
        | sed "s/.*listen_address:.*/listen_address: $i/" \
        | sed "s/.*rpc_address:.*/rpc_address: $i/" \
        | sed "s/.*rpc_interface:.*/#rpc_interface: $iface/" \
        | sed "s/.*listen_interface:.*/#listen_interface: $iface/" \
        > ${CASS_CONF}/cassandra-${i}.yaml
done

cat ${TEMPLATE_CASS_ENV_FILE} \
    | sed "s/jmxremote.authenticate=true/jmxremote.authenticate=false/" \
    >${CASS_ENV_FILE}

DBG $(ls -la ${CASS_YAML_FILE})
DBG $(ls -la ${CASS_ENV_FILE})

TIME_START=`date +"%T.%3N"`
echo "Launching Cassandra in the following hosts: $casslist"
export CASSANDRA_NODELIST=$(echo $casslist | sed -e 's+ +,+g')

DBG " CASSANDRA_NODELIST var: "$CASSANDRA_NODELIST

# Clearing data from previous executions and checking symlink coherence
run srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=1 --nodes=$N_NODES $MODULE_PATH/tmp-set.sh $DATA_HOME $COMM_HOME $SAV_CACHE $ROOT_PATH
sleep 5

if [ "$(cat $RECOVER_FILE)" != "" ]
then
    RECOVERTIME1=`date +"%T.%3N"`
    # Moving data to each datapath
    run srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=1 --nodes=$N_NODES $MODULE_PATH/recover.sh $ROOT_PATH $UNIQ_ID ${CASS_CONF}
    RECOVERTIME2=`date +"%T.%3N"`

    DBG "[DEBUG] Recover process initial datetime: $RECOVERTIME1"
    DBG "[DEBUG] Recover process final datetime: $RECOVERTIME2"

    show_time "[STATS] Cluster recover process (copy files and set tokens for all nodes) took: " $RECOVERTIME1 $RECOVERTIME2
fi       
# Launching Cassandra in every node

CASS_CORE_LIST=$( gen_mask LAST $C4S_CASSANDRA_CORES)

run srun --overlap --mem=0 --nodelist=$CASSANDRA_NODELIST \
	--ntasks=$N_NODES \
	--ntasks-per-node=1 \
	--nodes=$N_NODES \
	--cpu-bind=verbose,mask_cpu:${CASS_CORE_LIST} \
	$MODULE_PATH/cass_node.sh $UNIQ_ID &
sleep 5

if [ "X$STREAMING" != "X" ]; then
    if [ ${STREAMING,,} == "true" ]; then
        source $HECUBA_ROOT/bin/cassandra4slurm/launch_kafka.sh
        launch_kafka $CASSFILE $UNIQ_ID $N_NODES
    fi
fi

# Checking cluster status until all nodes are UP (or timeout)
firstnode=$(echo $seeds | awk -F ',' '{ print $1 }')
echo "Checking..."
RETRY_COUNTER=0
get_nodes_up $firstnode
while [ "$NODE_COUNTER" != "$N_NODES" ] && [ $RETRY_COUNTER -lt $RETRY_MAX ]; do
    echo "$NODE_COUNTER/$N_NODES nodes UP. Retry #$RETRY_COUNTER"
    echo "Checking..."
    sleep 5
    get_nodes_up $firstnode
    ((RETRY_COUNTER++))
done
if [ "$NODE_COUNTER" == "$N_NODES" ]
then
    TIME_END=`date +"%T.%3N"`
    echo "Cassandra Cluster with "$N_NODES" nodes started successfully."

    show_time "[STATS] Cluster launching process took: " $TIME_START $TIME_END
else
    echo "[STATS] ERROR: Cassandra Cluster RUN timeout. Check STATUS."
    exit_bad_node_status
fi


# THIS IS THE APPLICATION CODE EXECUTING SOME TASKS USING CASSANDRA DATA, ETC
echo "CHECKING CASSANDRA STATUS: "
$CASS_HOME/bin/nodetool -Dcom.sun.jndi.rmiURLParsing=legacy -h $firstnode status


CNAMES=$(sed ':a;N;$!ba;s/\n/,/g' $CASSFILE.ips)
export CONTACT_NAMES=$CNAMES
echo "CONTACT_NAMES=$CONTACT_NAMES"
#echo $CNAMES | tr , '\n' > $HOME/bla.txt # Set list of nodes (with interface) in PyCOMPSs file
PYCOMPSS_STORAGE=$C4S_HOME/pycompss_storage_"$UNIQ_ID".txt
echo $CNAMES | tr , '\n' > $PYCOMPSS_STORAGE # Set list of nodes (with interface) in PyCOMPSs file

# Workaround: Creating hecuba.istorage before execution.
#$CASS_HOME/bin/cqlsh $(head -n 1 $CASSFILE) < $MODULE_PATH/hecuba-istorage.cql
#$CASS_HOME/bin/cqlsh $(head -n 1 $CASSFILE) < $MODULE_PATH/tables_numpy.cql

check_cassandra_is_available $firstnode

source $MODULE_PATH/initialize_hecuba.sh $firstnode ${CASS_HOME}/bin/cqlsh

if [ "0$SCHEMA" != "0" ]; then
  echo "Connecting to $firstnode for tables creation. Schema $SCHEMA."
  $CASS_HOME/bin/cqlsh $firstnode -f $SCHEMA
  sleep 10
fi

# CASSANDRA UP AND RUNNING... SYNC WITH OTHER SCRIPTS...
cp $CASSFILE $CASSFILETOSYNC

# Application launcher
if [ "$APP_NODES" != "0" ]; then
    APP_AND_PARAMS=$(cat $APPPATHFILE)
    if [ "X$C4S_APP_CORES" == X ]; then
	#if it is not defined, we allocate equal number of cores per process
        if [ "$DISJOINT" == "1" ]; then
            # TODO: is this required??? --> export PYCOMPSS_NODES=$(cat $APPFILE | tr '\n' ',' | sed "s/,/$full_iface,/g" | rev | cut -c 2- | rev) # Workaround for disjoint executions with PyCOMPSs
	    APP_AVAIL_CORES=$(get_numphyscores)
        else
            APP_AVAIL_CORES=$(( $(get_numphyscores) - $C4S_CASSANDRA_CORES ))
        fi
	C4S_APP_CORES=$(( ($APP_AVAIL_CORES * $APP_NODES) / $APP_NTASKS ))
	if [ "$C4S_APP_CORES" == "0" ]; then
		echo " WARN: More tasks than available cores. Using 1 core per task."
		C4S_APP_CORES=1
	fi
    fi



    DBG "C4S_APP_CORES = $C4S_APP_CORES"
    if [ "$PYCOMPSS_APP" == "1" ]; then
        PYCOMPSS_FLAGS=$(cat $PYCOMPSS_FLAGS_FILE)
        # TODO: Check if escaping chars is needed for app parameters
        echo "export CONTACT_NAMES=$CNAMES" >> $ENVFILE # Setting Cassandra cluster environment variable for Hecuba

        C4S_COMPSS_CORES=$C4S_APP_CORES

        cat $MODULE_PATH/pycompss_template.sh \
            | sed "s+PLACEHOLDER_CASSANDRA_NODES_FILE+$CASSFILE.ips+g" \
            | sed "s+PLACEHOLDER_PYCOMPSS_NODES_FILE+$APPFILE+g" \
            | sed "s+PLACEHOLDER_APP_PATH_AND_PARAMETERS+$APP_AND_PARAMS+g" \
            | sed "s+PLACEHOLDER_PYCOMPSS_FLAGS+$PYCOMPSS_FLAGS+g" \
            | sed "s+PLACEHOLDER_PYCOMPSS_STORAGE+$PYCOMPSS_STORAGE+g" \
            > $PYCOMPSS_FILE

        APP_NODELIST=$(cat $APPFILE | tr '\n' ',')
	echo " TODO: THIS CODE IS NOT REVISED. It may fail "
	#TODO (is this required?) ---> SLURM_JOB_NUM_NODES=$APP_NODES SLURM_NTASKS=$(( $APP_NODES * $C4S_COMPSS_CORES )) SLURM_JOB_NODELIST=${APP_NODELIST::-1} bash $PYCOMPSS_FILE
        bash $PYCOMPSS_FILE
    else

	# CASSPIDFILE must point to the dirname containing the .pid file generated at cass_node.sh
        export CASSPIDFILE=$C4S_HOME/conf/${UNIQ_ID}/

        DBG " RUNNING IN $APP_NODES APP_NODES WITH NTASKS_PERNODE $APP_NTASKS"
	source $MODULE_PATH/app_node.sh $APP_NTASKS $UNIQ_ID $WITHDLB
    fi
else
    echo "[INFO] This job is not configured to run any application. Skipping..."
fi
# End of the application execution code

# Wait for a couple of minutes to assure that the data is stored
while [ "$(cat $C4S_HOME/stop."$UNIQ_ID".txt)" != "1" ]; do
    #echo "Sleeping until "$C4S_HOME"/stop."$UNIQ_ID".txt has value \"1\"."
    sleep 2
done

# If an snapshot was ordered, it is done
if [ "$(cat $SNAPSHOT_FILE)" == "1" ]
then 
    TIME1=`date +"%T.%3N"`
    SNAP_NAME="$THETIME"
    # Looping over the assigned hosts until the snapshots are confirmed
    DBG " Launching snapshot tasks on nodes $CASSANDRA_NODELIST"

    source $MODULE_PATH/snapshot.sh $SNAP_NAME $ROOT_PATH $CLUSTER $UNIQ_ID ${CASS_HOME}/bin/nodetool

    SNAP_CONT=0
    while [ "$SNAP_CONT" != "$N_NODES" ]
    do
        SNAP_CONT=0
        for u_host in $casslist
        do
            if [ -f $C4S_HOME/snap-status-$SNAP_NAME-$u_host-file.txt ]
            then
                SNAP_CONT=$(($SNAP_CONT+1))
            fi
        done
    done
    
    TIME2=`date +"%T.%3N"`
    DBG " Snapshot initial datetime: $TIME1"
    DBG " Snapshot final datetime: $TIME2"

    show_time "[STATS] Snapshot process took: " $TIME1 $TIME2

    # Cleaning status files
    rm -f $C4S_HOME/snap-status-$SNAP_NAME-*-file.txt
fi
sleep 10

DBG " Stopping CASSANDRA JOB"
run srun  --overlap --mem=0 --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --nodes=$N_NODES $MODULE_PATH/killer.sh
