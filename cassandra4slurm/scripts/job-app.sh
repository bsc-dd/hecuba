#!/bin/bash
#TO BE REMOVED#SBATCH --qos=debug
###############################################################################################################
#                                                                                                             #
#                                      Cassandra4Slurm App Job for Slurm                                      #
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

CASS_IFACE="-ib0"
iface=$(echo "$CASS_IFACE" | sed 's/-//g')

# Parameters
UNIQ_ID=${1}          # Unique ID to identify related files
PYCOMPSS_APP=${2}     # Application execution using PyCOMPSs. 0: No, 1: Yes
CLUSTER_ID=${3}       # Cassandra4Slurm cluster identificator
#$UNIQ_ID $APP_NODES $PYCOMPSS_SET $CLUSTER_ID

export C4S_HOME=$HOME/.c4s
MODULE_PATH=$HECUBA_ROOT/bin/cassandra4slurm
HECUBA_ENVIRON=$C4S_HOME/conf/hecuba_environment
CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg
export CASSFILE=$C4S_HOME/casslist-"$CLUSTER_ID".txt
export APPFILE=$C4S_HOME/applist-"$UNIQ_ID".txt
APPPATHFILE=$C4S_HOME/app-"$UNIQ_ID".txt
PYCOMPSS_FLAGS_FILE=$C4S_HOME/pycompss-flags-"$UNIQ_ID".txt
PYCOMPSS_FILE=$C4S_HOME/pycompss-"$UNIQ_ID".sh
PYCOMPSS_STORAGE=$C4S_HOME/pycompss-storage-"$UNIQ_ID".txt
scontrol show hostnames $SLURM_NODELIST > $APPFILE

CASSANDRA_NODES=$(cat $CASSFILE | wc -l)
APP_NODES=$(cat $APPFILE | wc -l)

THETIME=$(date "+%Y%m%dD%H%Mh%Ss")
source $MODULE_PATH/hecuba_debug.sh
DBG "  APP_NODES="$APP_NODES
DBG "  CASSANDRA_NODES="$CASSANDRA_NODES
DBG "  CLUSTER_ID="$CLUSTER_ID

CNAMES=$(sed ':a;N;$!ba;s/\n/,/g' $CASSFILE)$CASS_IFACE
CNAMES=$(echo $CNAMES | sed "s/,/$CASS_IFACE,/g")
export CONTACT_NAMES=$CNAMES
DBG "  CONTACT_NAMES=$CONTACT_NAMES"
#echo $CNAMES | tr , '\n' > $HOME/bla.txt # Set list of nodes (with interface) in PyCOMPSs file
echo $CNAMES | tr , '\n' > $PYCOMPSS_STORAGE # Set list of nodes (with interface) in PyCOMPSs file

# Workaround: Creating hecuba.istorage before execution.
#$CASS_HOME/bin/cqlsh $(head -n 1 $CASSFILE)$CASS_IFACE < hecuba-istorage.cql
#$CASS_HOME/bin/cqlsh $(head -n 1 $CASSFILE)$CASS_IFACE < $MODULE_PATH/tables_numpy.cql

# Application launcher
if [ "$APP_NODES" != "0" ]; then
    if [ "$PYCOMPSS_APP" == "1" ]; then
        APP_AND_PARAMS=$(cat $APPPATHFILE)
        PYCOMPSS_FLAGS=$(cat $PYCOMPSS_FLAGS_FILE)
        echo "export CONTACT_NAMES=$CNAMES" > ~/contact_names.sh # Setting Cassandra cluster environment variable for Hecuba
        if [ -f $HECUBA_ENVIRON ]; then
		cat $HECUBA_ENVIRON >> ~/contact_names.sh
	fi
        full_iface=$iface
        if [ "0$iface" != "0" ]; then
            full_iface="-"$iface
        fi
        DISJOINT=1 # Hardcoded for runapp executions
        if [ "$DISJOINT" == "1" ]; then
            export PYCOMPSS_NODES=$(cat $APPFILE | tr '\n' ',' | sed "s/,/$full_iface,/g" | rev | cut -c 2- | rev) # Workaround for disjoint executions with PyCOMPSs 
        fi
        cat $MODULE_PATH/pycompss_template.sh | sed "s+PLACEHOLDER_CASSANDRA_NODES_FILE+$CASSFILE+g" | sed "s+PLACEHOLDER_PYCOMPSS_NODES_FILE+$APPFILE+g" | sed "s+PLACEHOLDER_APP_PATH_AND_PARAMETERS+$APP_AND_PARAMS+g" | sed "s+PLACEHOLDER_PYCOMPSS_FLAGS+$PYCOMPSS_FLAGS+g" | sed "s+PLACEHOLDER_PYCOMPSS_STORAGE+$PYCOMPSS_STORAGE+g" > $PYCOMPSS_FILE
        bash $PYCOMPSS_FILE "-$iface" "$DISJOINT"  # Params - 1st: interface, 2nd: 1 to indicate disjoint execution
    else
        srun --ntasks-per-node=1 $MODULE_PATH/app_node.sh $UNIQ_ID
    fi
else
    # This will never be executed. Or something went really wrong.
    echo "[INFO] This job is not configured to run any application. Skipping..."
    echo "" > ~/contact_names.sh # Reset contact names to avoid re-using previous execution config
fi
# End of the application execution code
