#!/bin/bash
###############################################################################################################
#													      #
#                                 Application Node Launcher for Slurm clusters                                #
#                                          Eloy Gil - eloy.gil@bsc.es                                         #
#													      #
#                                        Barcelona Supercomputing Center                                      #
#		                                     .-.--_                                       	      #
#                    			           ,´,´.´   `.                                     	      #
#              			                   | | | BSC |                                     	      #
#                   			           `.`.`. _ .´                                     	      #
#                        		             `·`··                                         	      #
#													      #
###############################################################################################################
export C4S_HOME=$HOME/.c4s
APP_NTASKS=${1}
UNIQ_ID=${2}
WITHDLB=${3}
APP_PATH=$(cat $C4S_HOME/app-"$UNIQ_ID".txt)

if [[ "0$SCHEMA" != "0" ]]; then
#if [[ $APP_PATH = *"Alya"*  && -n ${HFETCH_ROOT} ]]; then
#  ALYA_KEYSPACE="${ALYA_KEYSPACE:-${SLURM_JOB_NAME}}"
#  PROBLEM="${PROBLEM:-${SLURM_JOBID}}"
#  SCHEMA=alyaqbeast.template
  node1=$(echo $CONTACT_NAMES | awk -F ',' '{print $1}')

#  sed ${HFETCH_ROOT}/tables_templates/$SCHEMA -e "s+PKSP+$ALYA_KEYSPACE+g" -e "s+PTAB+$PROBLEM+g" -e "s+QIDX+qb_${ALYA_KEYSPACE}_${PROBLEM}_xyz+g" > /tmp/tables.$SLURM_JOBID.txt

  echo "Connecting to $node1 for tables creation. Schema $SCHEMA."
  cqlsh $node1 -f $SCHEMA
#/tmp/tables.$SLURM_JOBID.txt
  sleep 10
fi


echo "Launching application: $APP_PATH"
if [ "$WITHDLB" == "1" ]; then
	DLB_HOME=/apps/GPP/PM/dlb/git/impi/
	preload="$DLB_HOME/lib/libdlb_mpi.so"
	preload="$preload:$HOME/dlb/doc/examples/lewi_custom_rt/libcustom_rt.so"
	export LD_PRELOAD=$preload
fi
# FOR SOME REASON the 'srun' tool does not take into account the SLURM Environment VARIABLES
BINDING="--cpu-bind=verbose"	# to show assigned cpu_mask
srun --mem=0 --nodes=$SLURM_JOB_NUM_NODES --ntasks=$APP_NTASKS $BINDING --cpus-per-task=$C4S_APP_CORES $APP_PATH

echo "[INFO] The application execution has stopped in node $(hostname)"
