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
PYCOMPSS_APP=${4}     # Application execution using PyCOMPSs. 0: No, 1: Yes
DISJOINT=${5}         # Guarantee disjoint allocation. 1: Yes, empty otherwise

export C4S_HOME=$HOME/.c4s
HECUBA_ENVIRON=$C4S_HOME/conf/hecuba_environment
MODULE_PATH=$HECUBA_ROOT/bin/cassandra4slurm
CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg
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
scontrol show hostnames $SLURM_NODELIST > $NODEFILE

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

if [ "X$HECUBA_ARROW" != "X" ]; then
    #Set HECUBA_ARROW_PATH to the DATA_PATH/TIME
    export HECUBA_ARROW_PATH=$ROOT_PATH/
    echo "export HECUBA_ARROW_PATH=$ROOT_PATH/" >> $ENVFILE
fi

if [ "$DISJOINT" == "1" ]; then
    C4S_CASSANDRA_CORES=$(nproc --all)
else
    C4S_CASSANDRA_CORES=4
fi

RETRY_MAX=3000 # This value is around 50, higher now to test big clusters that need more time to be completely discovered

# Generating nodefiles
tail -n $CASSANDRA_NODES $NODEFILE > $CASSFILE
head -n $APP_NODES $NODEFILE > $APPFILE
export APPNODELIST=$(cat $APPFILE | tr '\n' ',' | sed "s/,/$full_iface,/g" | rev | cut -c 2- | rev)

source $MODULE_PATH/hecuba_debug.sh

DBG " iter="$iter
DBG " app_count="$app_count
DBG " APP_NODES="$APP_NODES
DBG " CASSANDRA_NODES="$CASSANDRA_NODES
DBG " DISJOINT="$DISJOINT

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
    first_node=`head -n1 $CASSFILE`"$CASS_IFACE"
    NODE_COUNTER=$($CASS_HOME/bin/nodetool -h $first_node status | sed 1,5d | sed '$ d' | awk '{ print $1 }' | grep "UN" | wc -l)
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
DBG " I am $(hostname)."
#export REPLICA_FACTOR=2

# If template is not there, it is copied from cassandra config folder 
if [ ! -f $C4S_HOME/conf/template.yaml ]; then
    cp $CASS_HOME/conf/cassandra.yaml $C4S_HOME/conf/template.yaml
fi

# If original config exists and template does not, it is moved
if [ -f $CASS_HOME/conf/cassandra.yaml ] && [[ ! -s $C4S_HOME/conf/template.yaml ]]; then
    rm -f $C4S_HOME/conf/template.yaml
    cp $CASS_HOME/conf/cassandra.yaml $C4S_HOME/conf/template.yaml
fi

casslist=`cat $CASSFILE`
seedlist=`head -n 2 $CASSFILE`
seeds=`echo $seedlist | sed "s/ /-$iface,/g"`
seeds=$seeds-$iface #using only infiniband atm, will change later
sed "s/.*seeds:.*/          - seeds: \"$seeds\"/" $C4S_HOME/conf/template.yaml | sed "s/.*rpc_interface:.*/rpc_interface: $iface/" | sed "s/.*listen_interface:.*/listen_interface: $iface/" | sed "s/.*listen_address:.*/#listen_address: localhost/" | sed "s/.*rpc_address:.*/#rpc_address: localhost/" | sed "s/.*initial_token:.*/#initial_token:/" > $C4S_HOME/conf/template-aux-"$SLURM_JOB_ID".yaml


DBG $(ls -la $C4S_HOME/conf/template-aux-"$SLURM_JOB_ID".yaml)

TIME_START=`date +"%T.%3N"`
echo "Launching Cassandra in the following hosts: $casslist"
export CASSANDRA_NODELIST=$(echo $casslist | sed -e 's+ +,+g')

DBG " CASSANDRA_NODELIST var: "$CASSANDRA_NODELIST

# Clearing data from previous executions and checking symlink coherence
srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=1 --nodes=$N_NODES $MODULE_PATH/tmp-set.sh $CASS_HOME $DATA_HOME $COMM_HOME $SAV_CACHE $ROOT_PATH $CLUSTER $UNIQ_ID
sleep 5

if [ "$(cat $RECOVER_FILE)" != "" ]
then
    RECOVERTIME1=`date +"%T.%3N"`
    # Moving data to each datapath
    srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=1 --nodes=$N_NODES $MODULE_PATH/recover.sh $ROOT_PATH $UNIQ_ID
    RECOVERTIME2=`date +"%T.%3N"`

    DBG "[DEBUG] Recover process initial datetime: $RECOVERTIME1"
    DBG "[DEBUG] Recover process final datetime: $RECOVERTIME2"

    show_time "[STATS] Cluster recover process (copy files and set tokens for all nodes) took: " $RECOVERTIME1 $RECOVERTIME2
fi       
#exit # TODO QUITAR ESTO DE AQUÍ, SI NO NO FUNCIONA!!! TODO
# Launching Cassandra in every node

srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=$C4S_CASSANDRA_CORES --nodes=$N_NODES $MODULE_PATH/cass_node.sh $UNIQ_ID &
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

    show_time "[STATS] Cluster launching process took: " $TIME_START $TIME_END
else
    echo "[STATS] ERROR: Cassandra Cluster RUN timeout. Check STATUS."
    exit_bad_node_status
fi


# THIS IS THE APPLICATION CODE EXECUTING SOME TASKS USING CASSANDRA DATA, ETC
echo "CHECKING CASSANDRA STATUS: "
first_node=`head -n1 $CASSFILE`"$CASS_IFACE"
$CASS_HOME/bin/nodetool -h $first_node status

sleep 12
firstnode=$(echo $seeds | awk -F ',' '{ print $1 }')
CNAMES=$(sed ':a;N;$!ba;s/\n/,/g' $CASSFILE)$CASS_IFACE
CNAMES=$(echo $CNAMES | sed "s/,/$CASS_IFACE,/g")
export CONTACT_NAMES=$CNAMES
echo "CONTACT_NAMES=$CONTACT_NAMES"
#echo $CNAMES | tr , '\n' > $HOME/bla.txt # Set list of nodes (with interface) in PyCOMPSs file
PYCOMPSS_STORAGE=$C4S_HOME/pycompss_storage_"$UNIQ_ID".txt
echo $CNAMES | tr , '\n' > $PYCOMPSS_STORAGE # Set list of nodes (with interface) in PyCOMPSs file

# Workaround: Creating hecuba.istorage before execution.
#$CASS_HOME/bin/cqlsh $(head -n 1 $CASSFILE)$CASS_IFACE < $MODULE_PATH/hecuba-istorage.cql
#$CASS_HOME/bin/cqlsh $(head -n 1 $CASSFILE)$CASS_IFACE < $MODULE_PATH/tables_numpy.cql

source $MODULE_PATH/initialize_hecuba.sh $firstnode

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
    if [ "$PYCOMPSS_APP" == "1" ]; then
        PYCOMPSS_FLAGS=$(cat $PYCOMPSS_FLAGS_FILE)
        # TODO: Check if escaping chars is needed for app parameters
        echo "export CONTACT_NAMES=$CNAMES" >> $ENVFILE # Setting Cassandra cluster environment variable for Hecuba

        full_iface=$iface
        if [ "0$iface" != "0" ]; then
            full_iface="-"$iface
        fi
        if [ "$DISJOINT" == "1" ]; then
            export PYCOMPSS_NODES=$(cat $APPFILE | tr '\n' ',' | sed "s/,/$full_iface,/g" | rev | cut -c 2- | rev) # Workaround for disjoint executions with PyCOMPSs 
            C4S_COMPSS_CORES=$(nproc --all)
        else
            C4S_COMPSS_CORES=$(( $(nproc --all) - $C4S_CASSANDRA_CORES ))
        fi

        cat $MODULE_PATH/pycompss_template.sh | sed "s+PLACEHOLDER_CASSANDRA_NODES_FILE+$CASSFILE+g" | sed "s+PLACEHOLDER_PYCOMPSS_NODES_FILE+$APPFILE+g" | sed "s+PLACEHOLDER_APP_PATH_AND_PARAMETERS+$APP_AND_PARAMS+g" | sed "s+PLACEHOLDER_PYCOMPSS_FLAGS+$PYCOMPSS_FLAGS+g" | sed "s+PLACEHOLDER_PYCOMPSS_STORAGE+$PYCOMPSS_STORAGE+g" > $PYCOMPSS_FILE

        APP_NODELIST=$(cat $APPFILE | tr '\n' ',')
        SLURM_JOB_NUM_NODES=$APP_NODES SLURM_NTASKS=$(( $APP_NODES * $C4S_COMPSS_CORES )) SLURM_JOB_NODELIST=${APP_NODELIST::-1} bash $PYCOMPSS_FILE "-$iface" # Params - 1st: interface
    else
        DBG " RUNNING IN $APP_NODES APP_NODES WITH NTASKS_PERNODE $SLURM_NTASKS_PER_NODE, NTASKS $SLURM_NTASKS AND NPROCS $SLURM_NPROCS"
        SLURM_JOB_NUM_NODES=$APP_NODES source $MODULE_PATH/app_node.sh $UNIQ_ID
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

    source $MODULE_PATH/snapshot.sh $SNAP_NAME $ROOT_PATH $CLUSTER $UNIQ_ID

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

sacct --delimiter="," -pj ${SLURM_JOB_ID} | grep cass_node | awk -F ',' '{print $1}' | xargs scancel
