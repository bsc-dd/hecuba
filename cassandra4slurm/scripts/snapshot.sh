#!/bin/bash
###############################################################################################################
#                                                                                                             #
#                                  Cassandra Node Snapshot Launcher for Slurm                                 #
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

C4S_HOME=$HOME/.c4s
CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg
source $CFG_FILE
SNAP_NAME=${1}
ROOT_PATH=${2}
CLUSTER=${3}
UNIQ_ID=${4}
DATA_HOME=$ROOT_PATH/cassandra-data
CASSFILE=$C4S_HOME/casslist-"$UNIQ_ID".txt
RINGFILE=$C4S_HOME/ringfile-"$UNIQ_ID".txt
RINGDONE=$C4S_HOME/ringdone-"$UNIQ_ID".txt
HST_IFACE="-ib0" #interface configured in the cassandra.yaml file


casslist=`cat $CASSFILE`

echo " [I] Creating Snapshot directory $SNAP_PATH"
mkdir -p $SNAP_PATH

echo " [I] Repairing cassandra"
for u_host in $casslist
do
    $CASS_HOME/bin/nodetool -h ${u_host} repair
done

first_node=`head -n1 $CASSFILE`

echo " [I] Generating RingFile $RINGFILE"
rm -f $RINGFILE $RINGDONE
$CASS_HOME/bin/nodetool -h ${first_node} ring > $RINGFILE
echo "1" > $RINGDONE


echo " [I] Generating Snapshot $SNAP_NAME"
for u_host in $casslist
do
    $CASS_HOME/bin/nodetool -h ${u_host} snapshot -t $SNAP_NAME
    $CASS_HOME/bin/nodetool -h ${u_host} drain
done
echo " [I] Snapshot Done. Copying to GPFS"

sacct --delimiter="," -pj ${SLURM_JOB_ID} | grep cass_node | awk -F ',' '{print $1}' | xargs scancel

srun --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=$C4S_CASSANDRA_CORES --nodes=$N_NODES $MODULE_PATH/copy_snapshot.sh $SNAP_NAME $ROOT_PATH $CLUSTER $UNIQ_ID
