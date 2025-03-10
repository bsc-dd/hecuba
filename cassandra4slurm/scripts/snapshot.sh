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
SNAP_NAME=${1}
ROOT_PATH=${2}
CLUSTER=${3}
UNIQ_ID=${4}
NODETOOL_CMD="${5} -Dcom.sun.jndi.rmiURLParsing=legacy"
DATA_HOME=$ROOT_PATH/cassandra-data
CASSFILE=$C4S_HOME/casslist-"$UNIQ_ID".txt
RINGFILE=$C4S_HOME/ringfile-"$UNIQ_ID".txt
RINGDONE=$C4S_HOME/ringdone-"$UNIQ_ID".txt

NODETOOLFILE=$C4S_HOME/nodetool-"$UNIQ_ID".txt

source $HECUBA_ROOT/bin/cassandra4slurm/hecuba_debug.sh
CFG_FILE=$C4S_HOME/conf/${UNIQ_ID}/cassandra4slurm.cfg
source $CFG_FILE

casslist=`cat $CASSFILE`

DBG " Creating Snapshot directory $SNAP_PATH"
mkdir -p $SNAP_PATH

DBG "  Repairing cassandra"
for u_host in $casslist
do
    ${NODETOOL_CMD} -h ${u_host} repair >> $NODETOOLFILE
done

first_node=`head -n1 $CASSFILE`

DBG "  Generating RingFile $RINGFILE"
rm -f $RINGFILE $RINGDONE
${NODETOOL_CMD} -h ${first_node} ring > $RINGFILE
echo "1" > $RINGDONE

echo "[INFO] Generating Snapshot $SNAP_NAME"

for u_host in $casslist
do
    ${NODETOOL_CMD} -h ${u_host} snapshot -t $SNAP_NAME >> $NODETOOLFILE
    ${NODETOOL_CMD} -h ${u_host} drain >> $NODETOOLFILE
done
DBG "  Snapshot Done. Copying to GPFS"

run srun --overlap --mem=0 --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --ntasks-per-node=1 --cpus-per-task=$C4S_CASSANDRA_CORES --nodes=$N_NODES $MODULE_PATH/copy_snapshot.sh $SNAP_NAME $ROOT_PATH $CLUSTER $UNIQ_ID $SNAP_PATH
