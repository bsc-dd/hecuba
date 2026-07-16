#!/bin/bash
###############################################################################################################
#                                                                                                             #
#                                        Cassandra Node Manager for HPC                                       #
#                                       Juanjo Costa - jcosta@ac.upc.edu                                      #
#                                    Yolanda Becerra - yolanda.becerra@bsc.es                                 #
#                                                                                                             #
#                                        Barcelona Supercomputing Center                                      #
#                                                                                                             #
#		                                     .-.--_                                       	      #
#                    			           ,´,´.´   `.                                     	      #
#              			                   | | | BSC |                                     	      #
#                   			           `.`.`. _ .´                                     	      #
#                        		             `·`··                                         	      #
#													      #
###############################################################################################################

export C4S_HOME=$HOME/.c4s
export UNIQ_ID=${1}
CFG_FILE=$C4S_HOME/conf/${UNIQ_ID}/cassandra4slurm.cfg
HECUBA_ENVIRON=$C4S_HOME/conf/${UNIQ_ID}/hecuba_environment
source $HECUBA_ROOT/bin/cassandra4slurm/hecuba_debug.sh # to get get_node_ip
source $CFG_FILE    # To get CASS_IFACE
source $HECUBA_ENVIRON
HOSTNAMEIP=$(get_node_ip $(hostname) $CASS_IFACE)
export CASSPIDFILE=$C4S_HOME/conf/${UNIQ_ID}/cassandra-${HOSTNAMEIP}.pid
echo "cass_mngr: Waiting Cassandra writes PID @$(hostname)"
while [ ! -s $CASSPIDFILE ]; do
   sleep 1
done
echo "Starting Cassandra manager @$(hostname) for PID $(cat $CASSPIDFILE)" # <--- este echo sale y el pid es el de cassandra
#run ls  # <-- esto tambien lo veo: veo [DEBUG] ls  y luego el resultado de ejecutar ls
#run $HECUBA_ROOT/bin/cass_mgr $(cat $CASSPIDFILE) & # <--- esto no lo veo, ni el [DEBUG] comando ni el print que hace el cass_mgr justo al principio del main
#run EXTRAE_CONFIG_FILE=/home/bsc/bsc031226/hecuba-benchmarking/fake_fesom2/cass_mgr.xml $HECUBA_ROOT/bin/cass_mgr $(cat $CASSPIDFILE)  # <--- asi lo veo pero ahora salta otro error
run $HECUBA_ROOT/bin/cass_mgr $(cat $CASSPIDFILE)  # <--- asi lo veo pero ahora salta otro error
