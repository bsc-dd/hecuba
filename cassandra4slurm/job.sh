#!/bin/bash
#TO BE REMOVED#SBATCH --qos=debug
###############################################################################################################
#                                                                                                             #
#                                        Cassandra4Slurm Job for Slurm                                        #
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

iface=$(echo "$CASS_IFACE" | sed 's/-//g')

# Parameters
UNIQ_ID=${1}          # Unique ID to identify related files
CASSANDRA_NODES=${2}  # Number of Cassandra nodes to spawn
APP_NODES=${3}        # Number of nodes to run the application
PYCOMPSS_APP=${4}     # Application execution using PyCOMPSs. 0: No, 1: Yes
DISJOINT=${5}         # Guarantee disjoint allocation. 1: Yes, empty otherwise

export C4S_HOME=$HOME/.c4s
MODULE_PATH=/apps/HECUBA/0.1/lib/cassandra4slurm
CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg
NODEFILE=$C4S_HOME/hostlist-"$UNIQ_ID".txt
export CASSFILE=$C4S_HOME/casslist-"$UNIQ_ID".txt
export APPFILE=$C4S_HOME/applist-"$UNIQ_ID".txt
APPPATHFILE=$C4S_HOME/app-"$UNIQ_ID".txt
PYCOMPSS_FLAGS_FILE=$C4S_HOME/pycompss-flags-"$UNIQ_ID".txt
PYCOMPSS_FILE=$C4S_HOME/pycompss-"$UNIQ_ID".sh
SNAPSHOT_FILE=$C4S_HOME/cassandra-snapshot-file-"$UNIQ_ID".txt
RECOVER_FILE=$C4S_HOME/cassandra-recover-file-"$UNIQ_ID".txt
rm -f $NODEFILE $CASSFILE $APPFILE
scontrol show hostnames $SLURM_NODELIST > $NODEFILE

export CASS_HOME=$(cat $CFG_FILE | grep "CASS_HOME=" | sed 's/CASS_HOME=//g' | sed 's/"//g')
export DATA_PATH=$(cat $CFG_FILE | grep "DATA_PATH=" | sed 's/DATA_PATH=//g' | sed 's/"//g') 
export SNAP_PATH=$(cat $CFG_FILE | grep "SNAP_PATH=" | sed 's/SNAP_PATH=//g' | sed 's/"//g')
mkdir -p $SNAP_PATH
THETIME=$(date "+%Y%m%dD%H%Mh%Ss")"-$SLURM_JOB_ID"
ROOT_PATH=$DATA_PATH/$THETIME
DATA_HOME=$ROOT_PATH/c4j/cassandra-data
COMM_HOME=$ROOT_PATH/c4j/cassandra-commitlog
SAV_CACHE=$ROOT_PATH/c4j/saved_caches
CLUSTER=$THETIME
C4S_CASSANDRA_CORES=4
RETRY_MAX=50

# Generating nodefiles
cass_count=0
for node in $(cat $NODEFILE)
do
    if [ $cass_count -eq $CASSANDRA_NODES ]; then
        break
    else
        echo $node >> $CASSFILE
        ((cass_count++))
    fi
done

app_count=0
iter=0
touch $APPFILE
for node in $(cat $NODEFILE)
do
    if [ "$DISJOINT" == "1" ] && [ $iter -lt $CASSANDRA_NODES ]; then
        # If disjoint execution is enabled cassandra nodes are skipped
        ((iter++))
        echo "[DEBUG] Omitting node: "$node
        continue
    elif [ $app_count -eq $APP_NODES ]; then
        break
    else
        echo $node >> $APPFILE
        ((app_count++))
    fi
done
export APPNODELIST=$(cat $APPFILE | tr '\n' ',' | sed "s/,/$full_iface,/g" | rev | cut -c 2- | rev)

echo "[DEBUG] iter="$iter
echo "[DEBUG] app_count="$app_count
echo "[DEBUG] APP_NODES="$APP_NODES
echo "[DEBUG] CASSANDRA_NODES="$CASSANDRA_NODES
echo "[DEBUG] DISJOINT="$DISJOINT

N_NODES=$(cat $CASSFILE | wc -l)

if [ "$(cat $RECOVER_FILE)" != "" ]; then
    CLUSTER="$(cat $(find $SNAP_PATH -type f -name $(cat $RECOVER_FILE)-cluster.txt | head -n 1))"
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
    NODE_STATE_LIST=`$CASS_HOME/bin/nodetool status | sed 1,5d | sed '$ d' | awk '{ print $1 }'`
    if [ "$NODE_STATE_LIST" != "" ]
    then
        NODE_COUNTER=0
        for state in $NODE_STATE_LIST
        do  
            if [ $state == "UN" ]
            then
                ((NODE_COUNTER++))
            fi  
        done
    fi 
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
    RECOVERING=$(cat $RECOVER_FILE)
    echo "[INFO] Recovering snapshot: $RECOVERING"
fi

echo "STARTING UP CASSANDRA..."
echo "I am $(hostname)."
export REPLICA_FACTOR=2

sleep 10

# If template is not there, it is copied from cassandra config folder 
if [ ! -f $C4S_HOME/conf/template.yaml ]; then
    cp $CASS_HOME/conf/cassandra.yaml $C4S_HOME/conf/template.yaml
fi

# If original config exists and template does not, it is moved
if [ -f $CASS_HOME/conf/cassandra.yaml ] && [[ ! -s $C4S_HOME/conf/template.yaml ]]; then
    rm -f $C4S_HOME/conf/template.yaml
    cp $CASS_HOME/conf/cassandra.yaml $C4S_HOME/conf/template.yaml
fi

hostlist=`cat $CASSFILE`
seeds=`echo $hostlist | sed "s/ /-$iface,/g"`
seeds=$seeds-$iface #using only infiniband atm, will change later
sed "s/.*seeds:.*/          - seeds: \"$seeds\"/" $C4S_HOME/conf/template.yaml | sed "s/.*rpc_interface:.*/rpc_interface: $iface/" | sed "s/.*listen_interface:.*/listen_interface: $iface/" | sed "s/.*listen_address:.*/#listen_address: localhost/" | sed "s/.*rpc_address:.*/#rpc_address: localhost/" | sed "s/.*initial_token:.*/#initial_token:/" > $C4S_HOME/conf/template-aux.yaml
mv $C4S_HOME/conf/template-aux.yaml $C4S_HOME/conf/template.yaml

TIME_START=`date +"%T.%3N"`
echo "Launching in the following hosts: $hostlist"
export CASSANDRA_NODELIST=$(echo $hostlist | sed -e 's+ +,+g')

# Clearing data from previous executions and checking symlink coherence
srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=4 --nodes=$N_NODES $MODULE_PATH/tmp-set.sh $CASS_HOME $DATA_HOME $COMM_HOME $SAV_CACHE $ROOT_PATH $CLUSTER $UNIQ_ID
sleep 5

if [ "$(cat $RECOVER_FILE)" != "" ]
then
    RECOVERTIME1=`date +"%T.%3N"`
    # Moving data to each datapath
    #srun --ntasks=$TOTAL_NODES --ntasks-per-node=1 $MODULE_PATH/recover.sh $ROOT_PATH $UNIQ_ID
    srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=4 --nodes=$N_NODES $MODULE_PATH/recover.sh $ROOT_PATH $UNIQ_ID
    RECOVERTIME2=`date +"%T.%3N"`

    echo "[STATS] Recover process initial datetime: $RECOVERTIME1"
    echo "[STATS] Recover process final datetime: $RECOVERTIME2"

    MILL1=$(echo $RECOVERTIME1 | cut -c 10-12)
    MILL2=$(echo $RECOVERTIME2 | cut -c 10-12)
    TIMESEC1=$(date -d "$RECOVERTIME1" +%s)
    TIMESEC2=$(date -d "$RECOVERTIME2" +%s)
    TIMESEC=$(( TIMESEC2 - TIMESEC1 ))
    MILL=$(( MILL2 - MILL1 ))

    # Adjusting seconds if necessary
    if [ $MILL -lt 0 ]
    then
        MILL=$(( 1000 + MILL ))
        TIMESEC=$(( TIMESEC - 1 ))
    fi

    echo "[STATS] Cluster recover process (copy files and set tokens for all nodes) took: "$TIMESEC"s. "$MILL"ms."    
fi       

# Launching Cassandra in every node

echo "RUNNING srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=$C4S_CASSANDRA_CORES --nodes=$N_NODES $MODULE_PATH/cass_node.sh $UNIQ_ID"
srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=4 --nodes=$N_NODES $MODULE_PATH/cass_node.sh $UNIQ_ID &
sleep 5

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
$CASS_HOME/bin/nodetool status

sleep 12
firstnode=$(echo $hostlist | awk '{ print $1 }')
CNAMES=$(sed ':a;N;$!ba;s/\n/,/g' $CASSFILE)$CASS_IFACE
CNAMES=$(echo $CNAMES | sed "s/,/$CASS_IFACE,/g")
export CONTACT_NAMES=$CNAMES
echo "CONTACT_NAMES=$CONTACT_NAMES"
#echo $CNAMES | tr , '\n' > $HOME/bla.txt # Set list of nodes (with interface) in PyCOMPSs file
PYCOMPSS_STORAGE=$C4S_HOME/pycompss_storage_"$UNIQ_ID".txt
echo $CNAMES | tr , '\n' > $PYCOMPSS_STORAGE # Set list of nodes (with interface) in PyCOMPSs file

# Workaround: Creating hecuba.istorage before execution.
#$CASS_HOME/bin/cqlsh $(head -n 1 $CASSFILE)$CASS_IFACE < hecuba-istorage.cql
#$CASS_HOME/bin/cqlsh $(head -n 1 $CASSFILE)$CASS_IFACE < $MODULE_PATH/tables_numpy.cql

# Application launcher
if [ "$APP_NODES" != "0" ]; then
    if [ "$PYCOMPSS_APP" == "1" ]; then
        APP_AND_PARAMS=$(cat $APPPATHFILE)
        PYCOMPSS_FLAGS=$(cat $PYCOMPSS_FLAGS_FILE)
        # TODO: Check if escaping chars is needed for app parameters
        echo "export CONTACT_NAMES=$CNAMES" > ~/contact_names.sh # Setting Cassandra cluster environment variable for Hecuba
        full_iface=$iface
        if [ "0$iface" != "0" ]; then
            full_iface="-"$iface
        fi
        if [ "$DISJOINT" == "1" ]; then
            export PYCOMPSS_NODES=$(cat $APPFILE | tr '\n' ',' | sed "s/,/$full_iface,/g" | rev | cut -c 2- | rev) # Workaround for disjoint executions with PyCOMPSs 
        fi
        cat $MODULE_PATH/pycompss_template.sh | sed "s+PLACEHOLDER_CASSANDRA_NODES_FILE+$CASSFILE+g" | sed "s+PLACEHOLDER_PYCOMPSS_NODES_FILE+$APPFILE+g" | sed "s+PLACEHOLDER_APP_PATH_AND_PARAMETERS+$APP_AND_PARAMS+g" | sed "s+PLACEHOLDER_PYCOMPSS_FLAGS+$PYCOMPSS_FLAGS+g" | sed "s+PLACEHOLDER_PYCOMPSS_STORAGE+$PYCOMPSS_STORAGE+g" > $PYCOMPSS_FILE
        bash $PYCOMPSS_FILE "-$iface" # Params - 1st: interface
    else
        if [ "$DISJOINT" != "1" ]; then
            export SLURM_NTASKS_PER_NODE=$((SLURM_NTASKS_PER_NODE - C4S_CASSANDRA_CORES))
            export SLURM_NTASKS=$((SLURM_NTASKS_PER_NODE*APP_NODES))
            export SLURM_NPROCS=$SLURM_NTASKS
        fi
        export APP_NODES=$APP_NODES
        source $MODULE_PATH/app_node.sh $UNIQ_ID
    fi
else
    echo "[INFO] This job is not configured to run any application. Skipping..."
    echo "" > ~/contact_names.sh # Reset contact names to avoid re-using previous execution config
fi
# End of the application execution code

# Wait for a couple of minutes to assure that the data is stored
while [ "$(cat $C4S_HOME/stop."$UNIQ_ID".txt)" != "1" ]; do
    #echo "Sleeping until "$C4S_HOME"/stop."$UNIQ_ID".txt has value \"1\"."
    sleep 2
done

# Don't continue until the status is stable
#while [ "$NDT_STATUS" != "$($CASS_HOME/bin/nodetool status)" ]
#do
#    NDT_STATUS=$($CASS_HOME/bin/nodetool status)
#    #sleep 60
#    sleep 20
#done

# If an snapshot was ordered, it is done
if [ "$(cat $SNAPSHOT_FILE)" == "1" ]
then 
    TIME1=`date +"%T.%3N"`
    SNAP_NAME="$THETIME"
    # Looping over the assigned hosts until the snapshots are confirmed
    #srun --ntasks=$TOTAL_NODES --ntasks-per-node=1 bash $MODULE_PATH/snapshot.sh $SNAP_NAME $ROOT_PATH $CLUSTER $UNIQ_ID
    srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=4 --nodes=$N_NODES bash $MODULE_PATH/snapshot.sh $SNAP_NAME $ROOT_PATH $CLUSTER $UNIQ_ID

    SNAP_CONT=0
    while [ "$SNAP_CONT" != "$N_NODES" ]
    do
        SNAP_CONT=0
        for u_host in $hostlist
        do
            if [ -f $C4S_HOME/snap-status-$SNAP_NAME-$u_host-file.txt ]
            then
                SNAP_CONT=$(($SNAP_CONT+1))
            fi
        done
    done
    
    TIME2=`date +"%T.%3N"`

    echo "[STATS] Snapshot initial datetime: $TIME1"
    echo "[STATS] Snapshot final datetime: $TIME2" 

    MILL1=$(echo $TIME1 | cut -c 10-12)
    MILL2=$(echo $TIME2 | cut -c 10-12)
    TIMESEC1=$(date -d "$TIME1" +%s)
    TIMESEC2=$(date -d "$TIME2" +%s)
    TIMESEC=$(( TIMESEC2 - TIMESEC1 ))
    MILL=$(( MILL2 - MILL1 ))

    # Adjusting seconds if necessary
    if [ $MILL -lt 0 ]
    then
        MILL=$(( 1000 + MILL ))
        TIMESEC=$(( TIMESEC - 1 ))
    fi

    echo "[STATS] Snapshot process took: "$TIMESEC"s. "$MILL"ms."
    #echo "Snapshot process took: "$TIMESEC"s. "$MILL"ms." > stress/"$TEST_BASE_FILENAME"_0.log

    # Cleaning status files
    rm -f $C4S_HOME/snap-status-$SNAP_NAME-*-file.txt
fi
sleep 10
srun  --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 bash $MODULE_PATH/killer.sh

# Kills the job to shutdown every cassandra service
exit_killjob
