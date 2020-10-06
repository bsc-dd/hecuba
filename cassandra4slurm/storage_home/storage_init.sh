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
echo "RECEIVED ARGS ARE:"
echo $@
# Parameters
export UNIQ_ID=${1}          # Unique ID to identify related files (in this case the JOBID)
echo "[DEBUG] UNIQ_ID IS: "$UNIQ_ID
MASTER_NODE=${2}      # Master node name (same as 3rd parameter)
echo "[DEBUG] MASTER NODE IS: "$MASTER_NODE
WORKER_NODES=${4}     # Worker nodes list
echo "[DEBUG] WORKER NODES ARE: "$WORKER_NODES


if [ "x$WORKER_NODES" == "x" ]; then
        echo "[WARNING] Empty worker node lists, launching cassandra on the master node"
	WORKER_NODES=$MASTER_NODE
fi

CASSANDRA_NODES=$(echo $WORKER_NODES | wc -w)  # Number of Cassandra nodes to spawn (in this case in every node)
echo "[DEBUG] CASSANDRA NODE NUMBER IS: "$CASSANDRA_NODES
DISJOINT=0            # Guarantee disjoint allocation. 1: Yes, 0 or empty otherwise

if [ "$(echo "${5}" | sed 's/-//g')" == "infiniband" ]; then
    iface="ib0"
else
    iface="eth0"
fi

if [ "x$HECUBA_ROOT" == "x" ]; then
    HECUBA_ROOT=/apps/HECUBA/0.1.4_dev
fi

MAKE_SNAPSHOT=0
STORAGE_PROPS=${6}

FILE_TO_SET_ENV_VARS=${7}

echo "FILE TO SET VARS IS ${FILE_TO_SET_ENV_VARS}"
echo "[DEBUG] STORAGE PROPS FILE: "$STORAGE_PROPS
echo "[DEBUG] STORAGE PROPS CONTENT:"
cat $STORAGE_PROPS
source $STORAGE_PROPS

export C4S_HOME=$HOME/.c4s
export HECUBA_ENVIRON=$C4S_HOME/conf/hecuba_environment
export C4S_CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg
export MODULE_PATH=$HECUBA_ROOT/bin/cassandra4slurm

SNAPSHOT_FILE=$C4S_HOME/cassandra-snapshot-file-"$UNIQ_ID".txt
RECOVER_FILE=$C4S_HOME/cassandra-recover-file-"$UNIQ_ID".txt

#check if cassandra is already running and then just configure hecuba environment
# CONTACT_NAMES sould be set in STORAGE_PROPS file
grep -q -v ^# $STORAGE_PROPS | grep -q CONTACT_NAMES
if [ $? -eq 0 ]; then
        firstnode=$(echo $CONTACT_NAMES | awk -F ',' '{ print $1 }')
        source $MODULE_PATH/initialize_hecuba.sh  $firstnode

        echo "CONTACT_NAMES=$CONTACT_NAMES"
        echo "export CONTACT_NAMES=$CONTACT_NAMES" > ${FILE_TO_SET_ENV_VARS}
        echo "FILE TO EXPORT VARS IS  ${FILE_TO_SET_ENV_VARS}"
        if [ -f $HECUBA_ENVIRON ]; then
                cat $HECUBA_ENVIRON >> ${FILE_TO_SET_ENV_VARS}
                echo "FILE WITH HECUBA ENVIRON IS  ${HECUBA_ENVIRON}"
        fi
        cat ${FILE_TO_SET_ENV_VARS}
exit

fi

export THETIME=$(date "+%Y%m%dD%H%Mh%Ss")"-$SLURM_JOB_ID"
source $C4S_CFG_FILE
export CASS_HOME
export DATA_PATH
export SNAP_PATH

mkdir -p $SNAP_PATH
export ROOT_PATH=$DATA_PATH/$THETIME
export DATA_HOME=$ROOT_PATH/cassandra-data
COMM_HOME=$ROOT_PATH/cassandra-commitlog
SAV_CACHE=$ROOT_PATH/saved_caches
if [ "$DISJOINT" == "1" ]; then
    C4S_CASSANDRA_CORES=48
else
    C4S_CASSANDRA_CORES=4
fi
RETRY_MAX=500 # This value is around 50, higher now to test big clusters that need more time to be completely discovered

#check if the user wants to start from a stored snapshot
if [ "X$RECOVER" !=  "X" ]; then
        echo $RECOVER > $RECOVER_FILE
        export CLUSTER=$RECOVER
        RECOVERING=$CLUSTER
        echo "[INFO] Recovering snapshot: $RECOVERING"
else
        export CLUSTER=$THETIME
fi



NODEFILE=$C4S_HOME/hostlist-"$UNIQ_ID".txt
export CASSFILE=$C4S_HOME/casslist-"$UNIQ_ID".txt
#export APPFILE=$C4S_HOME/applist-"$UNIQ_ID".txt
APPPATHFILE=$C4S_HOME/app-"$UNIQ_ID".txt
PYCOMPSS_FLAGS_FILE=$C4S_HOME/pycompss-flags-"$UNIQ_ID".txt
PYCOMPSS_FILE=$C4S_HOME/pycompss-"$UNIQ_ID".sh
rm -f $NODEFILE $CASSFILE #$APPFILE
scontrol show hostnames $SLURM_NODELIST > $NODEFILE
# Generating nodefiles
#yolandab: this assumes that the n initial nodes are worker nodes. We have the variable worker nodes with the nodes separated by blanks, for the run command we need to use comma as the separator
#head -n $CASSANDRA_NODES $NODEFILE > $CASSFILE
CASSANDRA_NODELIST=$(echo $WORKER_NODES |sed s/\ /,/g)
echo $CASSANDRA_NODELIST |tr "," "\n" > $CASSFILE
NODETOOL_HOST=$(echo $CASSANDRA_NODELIST |cut -d"," -f1)-$iface

echo "[DEBUG] iter="$iter
echo "[DEBUG] app_count="$app_count
echo "[DEBUG] APP_NODES="$APP_NODES
echo "[DEBUG] CASSANDRA_NODES="$CASSANDRA_NODES
echo "[DEBUG] DISJOINT="$DISJOINT

export N_NODES=$CASSANDRA_NODES


function exit_bad_node_status () {
    # Exit after getting a bad node status.
    echo "Cassandra Cluster Status: ERROR"
    echo "It was expected to find $N_NODES nodes UP nodes, found "$NODE_COUNTER"."
    echo "Exiting..."
    exit
}

function get_nodes_up () {
    NODE_COUNTER=$($CASS_HOME/bin/nodetool -h $NODETOOL_HOST status | sed 1,5d | sed '$ d' | awk '{ print $1 }' | grep "UN" | wc -l)
}

if [ ! -f $CASS_HOME/bin/cassandra ]; then
    echo "ERROR: Cassandra executable is not placed where it was expected. ($CASS_HOME/bin/cassandra)"
    echo "Exiting..."
    exit
fi

if [ ! -f $C4S_HOME/stop."$UNIQ_ID".txt ]; then
    #we will end cassandra when the application ends
    #TODO: add a configuration to storage_props to keep it alive
    echo "1" > $C4S_HOME/stop."$UNIQ_ID".txt
fi

echo "STARTING UP CASSANDRA..."
echo "I am $(hostname)."
if [ $N_NODES -gt 1 ]; then
	export REPLICA_FACTOR=2
else
	export REPLICA_FACTOR=1
fi

# If template is not there, it is copied from cassandra config folder
if [ ! -f $C4S_HOME/conf/template.yaml ]; then
    cp $CASS_HOME/conf/cassandra.yaml $C4S_HOME/conf/template.yaml
fi

# If original config exists and template does not, it is moved
if [ -f $CASS_HOME/conf/cassandra.yaml ] && [[ ! -s $C4S_HOME/conf/template.yaml ]]; then
    rm -f $C4S_HOME/conf/template.yaml
    cp $CASS_HOME/conf/cassandra.yaml $C4S_HOME/conf/template.yaml
fi

seedlist=`head -n 2 $CASSFILE`
seeds=`echo $seedlist | sed "s/ /-$iface,/g"`
seeds=$seeds-$iface #using only infiniband atm, will change later
sed "s/.*seeds:.*/          - seeds: \"$seeds\"/" $C4S_HOME/conf/template.yaml | sed "s/.*rpc_interface:.*/rpc_interface: $iface/" | sed "s/.*listen_interface:.*/listen_interface: $iface/" | sed "s/.*listen_address:.*/#listen_address: localhost/" | sed "s/.*rpc_address:.*/#rpc_address: localhost/" | sed "s/.*initial_token:.*/#initial_token:/" > $C4S_HOME/conf/template-aux-"$SLURM_JOB_ID".yaml
ls -la $C4S_HOME/conf/template-aux-"$SLURM_JOB_ID".yaml

TIME_START=`date +"%T.%3N"`
echo "Launching Cassandra in the following hosts: $CASSANDRA_NODELIST"

# If a snapshot is needed

if [ "$MAKE_SNAPSHOT" == "1" ]; then
    echo "CASSANDRA_NODELIST=$CASSANDRA_NODELIST" > $SNAPSHOT_FILE
    echo "N_NODES=$N_NODES" >> $SNAPSHOT_FILE
    echo "THETIME=$THETIME" >> $SNAPSHOT_FILE
    echo "ROOT_PATH=$ROOT_PATH" >> $SNAPSHOT_FILE
    echo "CLUSTER=$CLUSTER" >> $SNAPSHOT_FILE
    echo "MODULE_PATH=$MODULE_PATH" >> $SNAPSHOT_FILE
fi

# Clearing data from previous executions and checking symlink coherence
echo "srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=4 --nodes=$N_NODES $MODULE_PATH/tmp-set.sh $CASS_HOME $DATA_HOME $COMM_HOME $SAV_CACHE $ROOT_PATH $CLUSTER $UNIQ_ID"
srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=4 --nodes=$N_NODES $MODULE_PATH/tmp-set.sh $CASS_HOME $DATA_HOME $COMM_HOME $SAV_CACHE $ROOT_PATH $CLUSTER $UNIQ_ID
sleep 5

# Recover snapshot if any

if [ "X$RECOVERING" != "X" ]; then
    echo "srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=1 --nodes=$N_NODES $MODULE_PATH/recover.sh $ROOT_PATH $UNIQ_ID"
    srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=1 --nodes=$N_NODES $MODULE_PATH/recover.sh $ROOT_PATH $UNIQ_ID
fi

# Launching Cassandra in every node
echo "RUNNING srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=$C4S_CASSANDRA_CORES --nodes=$N_NODES $MODULE_PATH/enqueue_cass_node.sh $UNIQ_ID"
srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=4 --nodes=$N_NODES $MODULE_PATH/enqueue_cass_node.sh $UNIQ_ID &
sleep 5

# Cleaning config template
rm -f $C4S_HOME/conf/template-aux-"$SLURM_JOB_ID".yaml

# Checking cluster status until all nodes are UP (or timeout)
echo "Checking..."
RETRY_COUNTER=0
get_nodes_up
while [ "$NODE_COUNTER" != "$N_NODES" ] && [ $RETRY_COUNTER -lt $RETRY_MAX ]; do
    echo "Retry #$RETRY_COUNTER"
    echo "Checking..."
    sleep 5
    get_nodes_up
    ((RETRY_COUNTER++))
done
if [ "$NODE_COUNTER" == "$N_NODES" ]
then
    TIME_END=`date +"%T.%3N"`
    echo "Cassandra Cluster with "$N_NODES" nodes started successfully."
    MILL1=$(echo $TIME_START | cut -c 10-12)
    MILL2=$(echo $TIME_END | cut -c 10-12)
    TIMESEC1=$(date -d "$TIME_START" +%s)
    TIMESEC2=$(date -d "$TIME_END" +%s)
    TIMESEC=$(( TIMESEC2 - TIMESEC1 ))
    MILL=$(( MILL2 - MILL1 ))

    # Adjusting seconds if necessary
    if [ $MILL -lt 0 ]
    then
        MILL=$(( 1000 + MILL ))
        TIMESEC=$(( TIMESEC - 1 ))
    fi

    echo "[STATS] Cluster launching process took: "$TIMESEC"s. "$MILL"ms."
else
    echo "[STATS] ERROR: Cassandra Cluster RUN timeout. Check STATUS."
    exit_bad_node_status
fi

# THIS IS THE APPLICATION CODE EXECUTING SOME TASKS USING CASSANDRA DATA, ETC
echo "CHECKING CASSANDRA STATUS: "
$CASS_HOME/bin/nodetool -h $NODETOOL_HOST status


firstnode=$(echo $seeds | awk -F ',' '{ print $1 }')

sleep 10

source $MODULE_PATH/initialize_hecuba.sh  $firstnode

CNAMES=$(sed ':a;N;$!ba;s/\n/,/g' $CASSFILE)
CNAMES=$(echo $CNAMES | sed "s/,/-$iface,/g")-$iface
export CONTACT_NAMES=$CNAMES
echo "CONTACT_NAMES=$CONTACT_NAMES"
echo "export CONTACT_NAMES=$CONTACT_NAMES" > ${FILE_TO_SET_ENV_VARS}
echo "FILE TO EXPORT VARS IS  ${FILE_TO_SET_ENV_VARS}"
if [ -f $HECUBA_ENVIRON ]; then
	cat $HECUBA_ENVIRON >> ${FILE_TO_SET_ENV_VARS}
	echo "FILE WITH HECUBA ENVIRON IS  ${HECUBA_ENVIRON}"
fi
cat ${FILE_TO_SET_ENV_VARS}
#srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=4 --nodes=$N_NODES "bash export CONTACT_NAMES=$CONTACT_NAMES" &
PYCOMPSS_STORAGE=$C4S_HOME/pycompss_storage_"$UNIQ_ID".txt
echo $CNAMES | tr , '\n' > $PYCOMPSS_STORAGE # Set list of nodes (with interface) in PyCOMPSs file
