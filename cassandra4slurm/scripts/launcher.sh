#!/bin/bash
###############################################################################################################
#                                                                                                             #
#                                      Cassandra Cluster Launcher for HPC                                     #
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

export C4S_HOME=$HOME/.c4s
export CASS_IFACE="-ib0"
MODULE_PATH=$HECUBA_ROOT/bin/cassandra4slurm
CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg
HECUBA_ENV=$C4S_HOME/conf/hecuba_environment
HECUBA_TEMPLATE_FILE=$MODULE_PATH/hecuba_environment.template

UNIQ_ID="c4s"$(echo $RANDOM | cut -c -5)
DEFAULT_NUM_NODES=4
DEFAULT_MAX_TIME="04:00:00"
RETRY_MAX=30
PYCOMPSS_SET=0
EXEC_NAME=$(echo ${0} | sed "s+/+ +g" | awk '{ print $NF }')
QUEUE=""

function set_workspace () {
    mkdir -p $C4S_HOME/logs
    mkdir -p $C4S_HOME/conf
    DEFAULT_DATA_PATH=/scratch/tmp
    DEFAULT_CASSANDRA=$HECUBA_ROOT/cassandra-d8tree
    echo "#This is a Cassandra4Slurm configuration file. Every variable must be set and use an absolute path." > $CFG_FILE
    echo "# LOG_PATH is the default log directory." >> $CFG_FILE
    echo "LOG_PATH=\"$HOME/.c4s/logs\"" >> $CFG_FILE
    echo "# DATA_PATH is a path to be used to store the data in every node. Using the SSD local storage of each node is recommended." >> $CFG_FILE
    echo "DATA_PATH=\"$DEFAULT_DATA_PATH\"" >> $CFG_FILE
    echo "CASS_HOME=\"$DEFAULT_CASSANDRA\"" >> $CFG_FILE
    echo "# SNAP_PATH is the destination path for snapshots." >> $CFG_FILE
    echo "SNAP_PATH=\"$DEFAULT_DATA_PATH/hecuba/snapshots\"" >> $CFG_FILE
}

if [ ! -f $CFG_FILE ]; then
    set_workspace
    echo "INFO: A default Cassandra4Slurm config has been generated. Adapt the following file if needed and try again:"
    echo "$CFG_FILE"
    return
fi

# Cleaning old executions (more than a month ago)
`find $C4S_HOME -maxdepth 1 -mtime +30 -type f | grep -v ".cfg" | xargs rm -f`
# Cleaning old jobs (not in the queuing system anymore)
C4S_JOBLIST=$C4S_HOME/joblist.txt
C4S_SCONTROL=$C4S_HOME/scontrol.txt
C4S_SQUEUE=$C4S_HOME/squeue.txt
scontrol show job > $C4S_SCONTROL
squeue > $C4S_SQUEUE
touch $C4S_JOBLIST $C4S_HOME/newjoblist.txt
if [ $(squeue | wc -l) -eq 1 ]; then
    rm -f $C4S_JOBLIST; touch $C4S_JOBLIST
else
    OLDIFS=$IFS; IFS=$'\n';
    for job_line in $(cat $C4S_JOBLIST); do
        job_id=$(echo $job_line | awk '{ print $1 }')
        if [ $(grep "$job_id " $C4S_SQUEUE | wc -l) -gt 0 ]; then
            echo "$job_line" >> $C4S_HOME/newjoblist.txt
        fi 
    done
    IFS=$OLDIFS
    mv $C4S_HOME/newjoblist.txt $C4S_JOBLIST
fi



function usage () {
    # Prints a help message
    echo "Usage: . $EXEC_NAME [ -h | RUN [ -s ] [ -f ] [ --disjoint ] [ -nC=cass_nodes ] [ -nA=app_nodes ] [ -nT=total_nodes ] [ --appl=PATH ARGS ] [ --pycompss=PARAMS ] [ -t=JOB_MAX_TIME ] [ --jobname=JOBNAME ] [ --logs=DIR ] | RECOVER [ -s ] | STATUS [ cluster_id ] | STOP [ -c=cluster_id ] | KILL [ -c=cluster_id ] ]"
    echo " "
    echo "IMPORTANT: The leading dot is needed since this launcher sets some environment variables."
    echo " "
    echo "       -h:"
    echo "       Prints this usage help."
    echo " "
    echo "       RUN [ -s ] [ -f ] [ --disjoint ] [ -nC=cass_nodes ] [ -nA=app_nodes ] [ -nT=total_nodes ] [ --appl=PATH ARGS ] [ --pycompss=PARAMS ] [ -t=JOB_MAX_TIME ] [ --jobname=JOBNAME ] [ --qos=debug ] [ --logs=DIR ]:"
    echo "       Starts a new Cassandra Cluster and perhaps executes an application, depending of many optional parameters."
    echo "       nT is the total number of nodes to be reserved. If it is not specified it will be $DEFAULT_NUM_NODES."
    echo "       nC is the number of Cassandra nodes, a subset of nT. If it is not specified a Cassandra will be launched in each of nT nodes."
    echo "       nA is the number of application nodes, a subset of nT. If --appl is not used it will be ignored. If it is not specified the application will run in each of nT nodes."
    echo "       If flag --disjoint is used, the application nodes will be different than Cassandra ones. (In this situation nT>=nC+nA)"
    echo "       If nT>nC+nA, nT nodes will be reserved but a warning will be shown, as some resources will be initially unused."
    echo "       To execute an application after the Cassandra cluster is up it is used --appl to set the path to the executable and its arguments, if any."
    echo "       If the application must be executed using PyCOMPSs the variable --pycompss should contain its PyCOMPSs parameters."
    echo "       Using -s it will save a snapshot after the execution."
    echo "       Using -t it will set the maximum time of the job to this value, with HH:MM:ss format. Default is 4 hours (04:00:00)."
    echo "       Using -f it will run the Cassandra cluster, the application (if set), take the snapshot (if set) and finish the execution automatically. Default is keep alive."
    echo "       Using --jobname=<JOBNAME> that will be the job name sent to the queue."
    echo "       Using --logs=<directory path> that will be the destination of the log files."
    echo "       Using --qos=debug it will run in the testing queue. It has some restrictions (a single job and 2h max.) so any higher requirements will be rejected by the queuing system."
    echo "       Everything passed using --constraint flags will be sent as it is to the job scheduler."
    echo " "
    echo "       RECOVER [ -s ]:"
    echo "       Shows a list of snapshots from previous Cassandra Clusters and restores the chosen one."
    echo "       Using -s it will save a new snapshot after the execution."
    echo "       Using -r=recover_id the user can specify the snapshot to restore directly."
    echo " "
    echo "       STOP [ -c=cluster_id ]:"
    echo "       If there is a cassandra cluster it is stopped, if there are many shows a list of cluster_id that can be stopped."
    echo " "
    echo "       STATUS [ -c=cluster_id ]:"
    echo "       If there is a cassandra cluster it shows its status, if there are many shows a list of cluster_id that can be used to request the status of one of them."
    echo " "
    echo "       KILL:"
    echo "       If a Cassandra Cluster is running, it is killed, aborting the process."
    echo " "
}

function set_utils_paths () {
    export JOBNAME=$UNIQ_ID
    SNAPSHOT_FILE=$C4S_HOME/cassandra-snapshot-file-"$UNIQ_ID".txt
    RECOVER_FILE=$C4S_HOME/cassandra-recover-file-"$UNIQ_ID".txt
    APP_PATH_FILE=$C4S_HOME/app-"$UNIQ_ID".txt
    PYCOMPSS_FLAGS_FILE=$C4S_HOME/pycompss-flags-"$UNIQ_ID".txt
    if [ "$FINISH" == "1" ]; then
        echo "1" > $C4S_HOME/stop."$UNIQ_ID".txt
    fi
    if [ "$PYCOMPSS_SET" == "1" ]; then
        echo $PYCOMPSS_APP > $PYCOMPSS_FLAGS_FILE
    fi
    if [ "0$APP" != "0" ]; then
        echo $APP > $APP_PATH_FILE
    fi
}

function get_job_info () {
    # Gets the ID of the job that runs the Cassandra Cluster
    #echo "JOBNAME: "$JOBNAME
    if [ "0$JOBNAME" != "0" ]; then
        JOB_ID=$(cat $C4S_JOBLIST | grep " $JOBNAME " | head -n 1 | awk '{ print $1 }')
        if [ "0$JOB_ID" == "0" ]; then
            echo "Jobname $JOBNAME not found, exiting..."
            exit
        fi
        #echo "JOB_ID: "$JOB_ID
        JOB_INFO=$(squeue | grep "$JOB_ID ")
        JOB_STATUS=$(echo $JOB_INFO | awk '{ print $5 }')
        #echo "JOB_STATUS: "$JOB_STATUS
        echo "JobID: "$JOB_ID" | Job Name: "$JOBNAME" | Job Status: "$JOB_STATUS
    fi
}

function get_cluster_node () {
    # Gets the ID of the first node
    NODE_ID=$(head -n 1 $C4S_HOME/hostlist-"$JOBNAME".txt)
}

function get_cluster_ips () {
    # Gets the IP of every node in the cluster
    NODE_IPS=$(ssh $NODE_ID "$CASS_HOME/bin/nodetool -h $NODE_ID$CASS_IFACE status" | awk '/Address/{p=1;next}{if(p){print $2}}')
}

function exit_no_cluster () {
    # Any Cassandra cluster is running. Exit.
    echo "There is no running Cassandra cluster. Exiting..."
    exit
}

function exit_bad_node_status () {
    # Exit after getting a bad node status. 
    echo "Cassandra Cluster Status: ERROR"
    echo "One or more nodes are not up (yet?) - It was expected to find ""$(cat $N_NODES_FILE | wc -l)"" UP nodes."
    echo "Exiting..."
    exit
}

function test_if_cluster_up () {
    # Checks if other Cassandra Cluster is running or enqueued, warning if it is happening
    if [ $(cat $C4S_JOBLIST | wc -l) -gt 0 ] 
    then
        echo "[WARN] Another Cassandra Cluster is in the queuing system. If the new launch is not aborted now it will continue in 5 seconds."
        squeue
        sleep 5
        echo "Launching..."
    fi
}

function get_nodes_up () {
    get_job_info
    if [ "$JOB_ID" != "" ]
    then
        if [ "$JOB_STATUS" == "R" ]
        then    
            get_cluster_node 
            #NODE_STATE_LIST=`ssh -q $NODE_ID "$CASS_HOME/bin/nodetool status" | sed 1,5d | sed '$ d' | awk '{ print $1 }'`
            NODE_STATE_LIST=`$CASS_HOME/bin/nodetool -h $NODE_ID$CASS_IFACE status 2> /dev/null | sed 1,5d | sed '$ d' | awk '{ print $1 }'`
            if [ "$NODE_STATE_LIST" != "" ]
            then
                NODE_COUNTER=0
                for state in $NODE_STATE_LIST
                do  
                    if [ $state != "UN" ]
                    then
                        RETRY_COUNTER=$(($RETRY_COUNTER+1))
                        break
                    else
                        NODE_COUNTER=$(($NODE_COUNTER+1))
                    fi
                done
            fi
        fi
    fi
}

function get_max_of_two () {
    if [ $1 -gt $2 ]; then
        echo $1
    else
        echo $2
    fi
}

function set_snapshot_value () {
    # Writes snapshot option into file
    if [ "$SNAPSH" == "-s" ]
    then
        echo "1" > $SNAPSHOT_FILE
    else
        echo "0" > $SNAPSHOT_FILE
    fi
}


function launch_arrow_helpers () {
    # Launch the 'arrow_helper' tool at each node in NODES, and leave their logs in LOGDIR
    NODES=$1
    LOGDIR=$2
    if [ ! -d $LOGDIR ]; then
        echo "INFO: Creating directory to store Arrow helper logs at [$LOGDIR]:"
        mkdir -p $LOGDIR
    fi
    ARROW_HELPER=$HECUBA_ROOT/src/hecuba_repo/build/arrow_helper
    ARROW_HELPER=$HECUBA_ROOT/bin/arrow_helper


    for i in $(cat $NODES); do
        echo "INFO: Launching Arrow helper at [$i] Log at [$LOGDIR/arrow_helper.$i.out]:"
        #ssh  $i $ARROW_HELPER >& $LOGDIR/arrow_helper.$i.out &
        ssh  $i $ARROW_HELPER $LOGDIR/arrow_helper.$i.out &
    done
    #echo "INFO: Launching Arrow helper at [$NODES]:"
	#CNAMES=$(sed ':a;N;$!ba;s/\n/,/g' $NODES)
	#CNAMES=$(echo $CNAMES | sed "s/,/$CASS_IFACE,/g")$CASS_IFACE
    #echo "INFO: Launching Arrow helper at [$CNAMES]:"
    #srun -s --nodelist $NODES --ntasks-per-node=1 --cpus-per-task=4 $ARROW_HELPER
}

for i in "$@"; do
case $i in
    -h|--help)
    usage
    return
    ;;
    run|RUN)
    ACTION="RUN"
    echo "Action is RUN."
    shift
    ;;
    recover|RECOVER)
    ACTION="RECOVER"
    shift
    ;;
    stop|STOP)
    ACTION="STOP"
    JOBNAME=""
    shift
    ;;
    status|STATUS)
    ACTION="STATUS"
    JOBNAME=""
    shift
    ;;
    kill|KILL)
    ACTION="KILL"
    JOBNAME=""
    shift
    ;;
    -s|--snapshot)
    SNAPSH="-s"
    shift
    ;;
    -d|--disjoint)
    DISJOINT="1"
    shift
    ;;
    -c=*|--cluster_id=*)
    JOBNAME="${i#*=}"
    echo "The jobname is: "$JOBNAME
    shift
    ;;
    -q=*|--qos=*)
    QUEUE="--qos=""${i#*=}"
    shift
    ;;
    -res=*|--reservation=*)
    CONSTRAINTS=$CONSTRAINTS" --reservation=""${i#*=}"
    shift
    ;;
    -con=*|--constraint=*)
    CONSTRAINTS=$CONSTRAINTS" --constraint=""${i#*=}"
    shift
    ;;
    -nT=*|--total_nodes=*)
    TOTAL_NODES="${i#*=}"
    shift
    ;;
    -nA=*|--app_nodes=*)
    APP_NODES="${i#*=}"
    shift
    ;;
    -nC=*|--cassandra_nodes=*)
    CASSANDRA_NODES="${i#*=}"
    shift
    ;;
    -a=*|--appl=*)
    APP="${i#*=}"
    shift
    ;;
    -p=*|--pycompss=*)
    path_to_executable=$(which launch_compss)
    if [ "0$path_to_executable" == "0" ] ; then
        echo "[ERROR] COMPSs is not available. Remember to load COMPSs before executing an application. Exiting..."
        exit
    fi
    PYCOMPSS_APP="${i#*=}"
    PYCOMPSS_SET=1
    shift
    ;;
    -t=*|--time=*)
    JOB_MAX_TIME="${i#*=}"
    shift
    ;;
    -f|--finish)
    FINISH=1
    shift
    ;;
    -r=*|--recover_id=*)
    input_snap="${i#*=}"
    shift
    ;;
    -j=*|--jobname=*)
    UNIQ_ID="${i#*=}"
    UNIQ_ID=$(echo $UNIQ_ID | sed 's+ ++g')
    if [ $(grep " $UNIQ_ID " $C4S_HOME/joblist.txt | wc -l) -gt 0 ]; then
        #echo "Jobname "$UNIQ_ID" already in use. Continue? (y/n) "
        echo "Jobname "$UNIQ_ID" already in use. Aborting..."
        #read input_jobname
        #while [ "$input_jobname" != "y" ] && [ "$input_jobname" != "n" ]; do
        #    echo "Wrong option. Continue? (y/n) "
        #    read input_jobname
        #done
        #if [ "$input_jobname" == "n" ]; then
        #    echo "Aborted."
        #    exit
        #fi
    fi
    shift
    ;;
    -l=*|--logs=*)
    LOGS_DIR="${i#*=}"
    mkdir -p $LOGS_DIR
    shift
    ;;
    -r=*|--restore=*)
    input_snap="${i#*=}"
    shift
    ;;
    *)
    UNK_FLAGS=$UNK_FLAGS"${i#=*}"" "
    ;;
esac
done

if [ "0$UNK_FLAGS" != "0" ]; then
    if [ "$(echo $UNK_FLAGS | wc -w)" -gt 1 ]; then
        MANY_FLAGS="s"
    fi
    echo "ERROR: Unknown flag$MANY_FLAGS: "$UNK_FLAGS
    echo "Check help: $EXEC_NAME -h"
    exit
fi


if [ ! -f $HECUBA_ENV ]; then
    echo "[INFO] Environment variables to load NOT found at $HECUBA_ENV"
    echo "[INFO] Copying default values."
    cp $HECUBA_TEMPLATE_FILE $HECUBA_ENV
else
    echo "[INFO] Environment variables to load found at $HECUBA_ENV"
fi

# Feel free to change this to a "source" if you want, but I don't recommend it.
source $CFG_FILE
#CASS_HOME=$(cat $CFG_FILE | grep -v "#" | grep "CASS_HOME=" | tail -n 1 | sed 's/CASS_HOME=//g' | sed 's/"//g' | sed "s/'//g")
#SNAP_PATH=$(cat $CFG_FILE | grep -v "#" | grep "SNAP_PATH=" | tail -n 1 | sed 's/SNAP_PATH=//g' | sed 's/"//g' | sed "s/'//g")
# Feel free to change this to a "source" if you want, but I don't recommend it.
source $HECUBA_ENV

if [ ! -f $CASS_HOME/bin/cassandra ]; then
    echo "ERROR: Cassandra binary is not where it was expected. ($CASS_HOME/bin/cassandra)"
    echo "Edit the following file to continue: $CFG_FILE"
    exit
fi

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
if [ "$ACTION" == "RUN" ]; then
    # Updates UNIQ_ID related paths and files if needed
    set_utils_paths
    test_if_cluster_up
    if [ "0$APP" == "0" ]; then
        # If there is no application path, everything is simpler: nA and --disjoint flags are ignored in any case.
        APP_NODES=0
        if [ "0$TOTAL_NODES" == "0" ] && [ "0$CASSANDRA_NODES" != "0" ]; then
            # If nT is undefined but nC is not, they are equal
            TOTAL_NODES=$CASSANDRA_NODES
        elif [ "0$TOTAL_NODES" != "0" ] && [ "0$CASSANDRA_NODES" == "0" ]; then
            # If nC is undefined but nT is not, they are equal
            CASSANDRA_NODES=$TOTAL_NODES
        elif [ "0$TOTAL_NODES" != "0" ] && [ "0$CASSANDRA_NODES" != "0" ]; then
            if [ $CASSANDRA_NODES -gt $TOTAL_NODES ]; then
                # If nC > nT show an error and exit.
                echo "ERROR: The number of Cassandra nodes (-nC=$CASSANDRA_NODES) is bigger than the total number of nodes (-nT=$TOTAL_NODES). Aborting..."
                exit
            elif [ $CASSANDRA_NODES -lt $TOTAL_NODES ]; then
                # If nC < nT show a warning (unused nodes)
                echo "WARN: A total of $TOTAL_NODES will be reserved but only $CASSANDRA_NODES are used for Cassandra."
            fi
        else
            # Nothing is defined, default values apply
            TOTAL_NODES=$DEFAULT_NUM_NODES
            CASSANDRA_NODES=$DEFAULT_NUM_NODES
        fi
    elif [ "0$TOTAL_NODES" == "0" ]; then
        # If nT is not defined...
        if [ "0$APP_NODES" == "0" ] && [ "0$CASSANDRA_NODES" != "0" ]; then
            if [ "$DISJOINT" == "1" ]; then
                # If nT is not defined,  the --disjoint flag is used and (APP_NODES or CASSANDRA_NODES) is not defined: ERROR
                echo "ERROR: If the --disjoint flag is used both Cassandra nodes (nC) and application nodes (nA) must be set or at least one of them and the total (nT). Aborting..."
                exit
            else
                # If nT is not defined, the --disjoint flag is NOT used and (APP_NODES or CASSANDRA_NODES) is not defined: They are equal.
                APP_NODES=$CASSANDRA_NODES
                TOTAL_NODES=$CASSANDRA_NODES
            fi
        elif [ "0$APP_NODES" != "0" ] && [ "0$CASSANDRA_NODES" == "0" ]; then
            if [ "$DISJOINT" == "1" ]; then
                # If nT is not defined,  the --disjoint flag is used and (APP_NODES or CASSANDRA_NODES) is not defined: ERROR
                echo "ERROR: If the --disjoint flag is used both Cassandra nodes (nC) and application nodes (nA) must be set or at least one of them and the total (nT). Aborting..."
                exit
            else
                # If nT is not defined, the --disjoint flag is NOT used and (APP_NODES or CASSANDRA_NODES) is not defined: They are equal.
                CASSANDRA_NODES=$APP_NODES
                TOTAL_NODES=$APP_NODES
            fi
        elif [ "0$APP_NODES" != "0" ] && [ "0$CASSANDRA_NODES" != "0" ]; then
            if [ "$DISJOINT" == "1" ]; then
                # If nT is not set and both nA and nC are defined and the --disjoint flag is used, nT=nA+nC
                TOTAL_NODES=$((APP_NODES+CASSANDRA_NODES))
            else
                # If nT is not set and both nA and nC are defined and the --disjoint flag is NOT used, nT=max(nA,nC)
	        TOTAL_NODES=$(get_max_of_two $APP_NODES $CASSANDRA_NODES)
            fi
        else
            # When all nT, nC, nA are undefined, default values apply
            CASSANDRA_NODES=$DEFAULT_NUM_NODES
            APP_NODES=$DEFAULT_NUM_NODES
            if [ "$DISJOINT" == "1" ]; then
                # If even without any node number value set, the --disjoint flag is used, nT=2*DEFAULT_NUM_NODES
                TOTAL_NODES=$((DEFAULT_NUM_NODES+DEFAULT_NUM_NODES))
            else
                TOTAL_NODES=$DEFAULT_NUM_NODES
            fi
        fi
    else
        # If nT is defined...
        if [ "0$APP_NODES" == "0" ] && [ "0$CASSANDRA_NODES" != "0" ]; then
            if [ "$DISJOINT" == "1" ]; then
                # If nT is defined, the --disjoint flag is used and APP_NODES is not defined: nA=nT-nC
                APP_NODES=$((TOTAL_NODES-CASSANDRA_NODES))
                if [ $APP_NODES -lt 1 ]; then
                    echo "ERROR: There are not enough nodes to execute the application with the requested setup ($TOTAL_NODES total nodes, using $CASSANDRA_NODES exclusively for Cassandra)" 
                    echo "Check help with $EXEC_NAME -h for more info. Aborting..."
                    exit
                fi
            else
                # If nT is defined, the --disjoint flag is NOT used and APP_NODES is not defined: nA=nC
                APP_NODES=$CASSANDRA_NODES
                if [ $TOTAL_NODES -lt $CASSANDRA_NODES ]; then
                    echo "ERROR: There are not enough nodes to execute the application with the requested setup ($TOTAL_NODES total nodes, needing $CASSANDRA_NODES for Cassandra)"
                    echo "Check help with $EXEC_NAME -h for more info. Aborting..."
                    exit
                elif [ "$TOTAL_NODES" != "$CASSANDRA_NODES" ]; then
                    echo "WARN: A total of $TOTAL_NODES will be reserved but only $CASSANDRA_NODES are being shared for Cassandra and running the application."
                fi  
            fi  
        elif [ "0$APP_NODES" != "0" ] && [ "0$CASSANDRA_NODES" == "0" ]; then
            if [ "$DISJOINT" == "1" ]; then
                # If nT is defined, the --disjoint flag is used and CASSANDRA_NODES is not defined: nC=nT-nA
                CASSANDRA_NODES=$((TOTAL_NODES-APP_NODES))
                if [ $CASSANDRA_NODES -lt 1 ]; then
                    echo "ERROR: There are not enough nodes to execute the application with the requested setup ($TOTAL_NODES total nodes, using $APP_NODES exclusively for the application)" 
                    echo "Check help with $EXEC_NAME -h for more info. Aborting..."
                    exit
                fi
            else
                # If nT is defined, the --disjoint flag is NOT used and CASSANDRA_NODES is not defined: nT>=nC=nA.
                CASSANDRA_NODES=$APP_NODES
                if [ $TOTAL_NODES -lt $APP_NODES ]; then
                    echo "ERROR: There are not enough nodes to execute the application with the requested setup ($TOTAL_NODES total nodes, needing $APP_NODES for the application)"
                    echo "Check help with $EXEC_NAME -h for more info. Aborting..."
                    exit
                elif [ "$TOTAL_NODES" != "$CASSANDRA_NODES" ]; then
                    echo "WARN: A total of $TOTAL_NODES will be reserved but only $APP_NODES are being shared for Cassandra and running the application."
                fi  
            fi  
        elif [ "0$APP_NODES" != "0" ] && [ "0$CASSANDRA_NODES" != "0" ]; then
            if [ "$DISJOINT" == "1" ]; then
                # If all nT, nA and nC are defined and the --disjoint flag is used, it is checked that nT>=nC+nA
                SUM_NODES=$((APP_NODES+CASSANDRA_NODES))
                if [ $TOTAL_NODES -lt $SUM_NODES ]; then
                    # Not enough total resources, error.
                    echo "ERROR: There are not enough nodes to execute the application with the requested setup ($TOTAL_NODES total nodes, needing $APP_NODES for the application, and $CASSANDRA_NODES more for Cassandra)"
                    echo "Check help with $EXEC_NAME -h for more info. Aborting..."
                    exit
                elif [ $TOTAL_NODES -gt $SUM_NODES ]; then
                    # Reserving more nodes than needed, warning.
                    echo "WARN: A total of $TOTAL_NODES will be reserved but only $CASSANDRA_NODES are used for Cassandra and $APP_NODES more for the application."
                fi
            else
                # If nT is set and both nA and nC are defined and the --disjoint flag is NOT used, nT>=max(nA,nC)
                MAX_NODES=$(get_max_of_two $APP_NODES $CASSANDRA_NODES)
                if [ $TOTAL_NODES -lt $MAX_NODES ]; then
                    # Not enough total resources, error.
                    echo "ERROR: There are not enough nodes to execute the application with the requested setup ($TOTAL_NODES total nodes, needing $APP_NODES to be shared between the application and Cassandra)"
                    echo "Check help with $EXEC_NAME -h for more info. Aborting..."
                    exit
                elif [ "$TOTAL_NODES" -gt $MAX_NODES ]; then
                    #Reserving more nodes than needed, warning.
                    echo "WARN: A total of $TOTAL_NODES will be reserved but only $CASSANDRA_NODES are used for Cassandra and the application."
                fi
            fi
        else
            # When nT is defined but nC and nA are undefined
            CASSANDRA_NODES=$DEFAULT_NUM_NODES
            APP_NODES=$DEFAULT_NUM_NODES
            if [ "$DISJOINT" == "1" ]; then
                # If only nT is defined and --disjoint is used, nT MUST be divisible by 2 (to assign half to Cassandra and other half to the application)
                if (( $TOTAL_NODES % 2 == 0 )); then
                    CASSANDRA_NODES=$((TOTAL_NODES/2))
                    APP_NODES=$CASSANDRA_NODES
                else
                    # nT is not divisible by 2, show error.
                    echo "ERROR: If only the total number is set and --disjoint is used, nT must be divisible by two. Aborting..."
                    exit
                fi
            else
                # If only nT is defined, all nodes are shared. nT=nC=nA
                CASSANDRA_NODES=$TOTAL_NODES
                APP_NODES=$TOTAL_NODES
            fi
        fi
    fi
 
    #WHILE DEBUGGING
    echo "EXECUTION SUMMARY:"
    echo "# of Cassandra nodes: "$CASSANDRA_NODES
    echo "# of application nodes: "$APP_NODES
    echo "# total of requested nodes: "$TOTAL_NODES
    #END DEBUGGING
    
    echo "Job allocation started..."

    # Since this is a fresh launch, it assures that the recover file is empty
    echo "" > $RECOVER_FILE

    # Enables/Disables the snapshot option after the execution
    set_snapshot_value
    if [ "0$JOB_MAX_TIME" == "0" ]; then
        JOB_MAX_TIME=$DEFAULT_MAX_TIME
    fi
    echo "SUBMITTING sbatch --job-name="$UNIQ_ID" --nodes=$TOTAL_NODES --time=$JOB_MAX_TIME --exclusive --output=$LOGS_DIR/cassandra-%j.out --error=$LOGS_DIR/cassandra-%j.err $QUEUE $CONSTRAINTS $MODULE_PATH/job.sh $UNIQ_ID $CASSANDRA_NODES $APP_NODES $PYCOMPSS_SET $DISJOINT"

    SUBMIT_MSG=$(sbatch --job-name="$UNIQ_ID" --nodes=$TOTAL_NODES --time=$JOB_MAX_TIME --exclusive --output=$LOGS_DIR/cassandra-%j.out --error=$LOGS_DIR/cassandra-%j.err $QUEUE $CONSTRAINTS $MODULE_PATH/job.sh $UNIQ_ID $CASSANDRA_NODES $APP_NODES $PYCOMPSS_SET $DISJOINT) #nproc result is 48 here
    echo $SUBMIT_MSG" ("$UNIQ_ID")"
    JOB_NUMBER=$(echo $SUBMIT_MSG | awk '{ print $NF }')
    echo $JOB_NUMBER $UNIQ_ID" " >> $C4S_JOBLIST
    echo "Please, be patient. It may take a while until it shows a correct status (and it may show some harmless errors during this process)."
    RETRY_COUNTER=0
    sleep 15

    while [ "$NODE_COUNTER" != "$CASSANDRA_NODES" ] && [ $RETRY_COUNTER -lt $RETRY_MAX ]; do
        echo "Checking..."
        sleep 10
	get_nodes_up
    done
    if [ "$NODE_COUNTER" == "$CASSANDRA_NODES" ]
    then
	while [ ! -f "$C4S_HOME/casslist-"$UNIQ_ID".txt" ]; do
            sleep 3
	done
	sleep 3
        echo "Cassandra Cluster with "$CASSANDRA_NODES" node(s) started successfully."
	CNAMES=$(sed ':a;N;$!ba;s/\n/,/g' $C4S_HOME/casslist-"$UNIQ_ID".txt)$CASS_IFACE
	CNAMES=$(echo $CNAMES | sed "s/,/$CASS_IFACE,/g")
	export CONTACT_NAMES=$CNAMES
	echo "Contact names environment variable (CONTACT_NAMES) should be set to: $CNAMES"
    else
        echo "ERROR: Cassandra Cluster RUN timeout. Check STATUS."
    fi 
    launch_arrow_helpers $C4S_HOME/casslist-"$UNIQ_ID".txt $LOGS_DIR/$UNIQ_ID

elif [ "$ACTION" == "STATUS" ] || [ "$ACTION" == "status" ]
then
    # If there is a running Cassandra Cluster it prints the information of the nodes
    if [ "0$JOBNAME" == "0" ]; then
        if [ $(cat $C4S_JOBLIST | wc -l) -gt 1 ]; then
            echo "There are many Cassandra clusters running, specify which one do you want to check."
            echo "e.g. $EXEC_NAME STATUS c4s1337"
            squeue
            exit
        elif [ $(cat $C4S_JOBLIST | wc -l) -eq 1 ]; then
            JOBNAME=$(cat $C4S_JOBLIST | awk '{ print $NF }')
        else
            exit_no_cluster
        fi
    fi
    get_job_info
    if [ "$JOB_ID" != "" ]
    then
    	if [ "$JOB_STATUS" == "PD" ]
        then
            echo "The job is still pending. Wait for a while and try again."
            exit
        fi
        N_NODES=$(cat $C4S_HOME/casslist-"$JOBNAME".txt | wc -l)
        get_cluster_node 
        #NODE_STATE_LIST=`ssh $NODE_ID "$CASS_HOME/bin/nodetool -h $NODE_ID$CASS_IFACE status" | sed 1,5d | sed '$ d' | awk '{ print $1 }'`
        NODE_STATE_LIST=`$CASS_HOME/bin/nodetool -h $NODE_ID$CASS_IFACE status 2> /dev/null | sed 1,5d | sed '$ d' | awk '{ print $1 }'`
	if [ "$NODE_STATE_LIST" == "" ]
	then
            echo "ERROR: No status found. The Cassandra Cluster may be still bootstrapping. Try again later."
            exit
        fi
        NODE_COUNTER=0
        for state in $NODE_STATE_LIST
        do
            if [ $state != "UN" ]
            then
                echo "E1"
                exit_bad_node_status
            else
                NODE_COUNTER=$(($NODE_COUNTER+1))
            fi
       	done
# TODO: Fix N_NODES_FILE for this case, it will fail because a new process id is used to get the UNIQ_ID and is not matching with any host file.
# This check should look for cassandra clusters (if any) and show a list of identifiers to choose one / details of each of them or about the cluster if there is only one.
        NODES_FILE=$C4S_HOME/casslist-"$(squeue | grep $JOBNAME | awk '{ print $3 }')".txt
        if [ "$N_NODES" == "$NODE_COUNTER" ]
        then
            echo "Cassandra Cluster Status: OK"
       	    $CASS_HOME/bin/nodetool -h $NODE_ID$CASS_IFACE status 2> /dev/null
        else
            echo "E2"
            echo "NODES_EXPECTED: "$N_NODES
            echo "NODE_COUNTER: "$NODE_COUNTER
            exit_bad_node_status
        fi
    else
        exit_no_cluster
    fi
elif [ "$ACTION" == "RECOVER" ] || [ "$ACTION" == "recover" ]
then
    # Updates UNIQ_ID related paths and files if needed
    set_utils_paths
    test_if_cluster_up
    # Launches a new Cluster to recover a previous snapshot
    SNAP_LIST=$(ls -tr $SNAP_PATH)
    if [ "0$input_snap" == "0" ]; then 
        if [ "$SNAP_LIST" == "" ]
        then
            echo "There are no available snapshots to restore."
            echo "Exiting..."
            exit
        else
            echo "The following snapshots are available to be restored:"
        fi
        for snap in $SNAP_LIST
        do
            echo -e $snap
        done
        echo "Introduce a snapshot to restore: "
        read input_snap
        while [ "$(echo $SNAP_LIST | grep -w $input_snap)" == "" ]; do
            echo "ERROR: Wrong snapshot input. Introduce a snapshot to restore: "
            read input_snap
        done
    else
        if [ "$(echo $SNAP_LIST | grep -w $input_snap)" == "" ]; then
            echo "ERROR: Snapshot <<"$input_snap">> not available. Aborting..."
            exit
        fi
    fi
    N_NODES=$(find $SNAP_PATH/$input_snap -type f -name $input_snap-ring.txt | wc -l)

    # Set snapshot name to recover into file
    echo $input_snap > $RECOVER_FILE

    # Enables/Disables the snapshot option after the execution
    set_snapshot_value

    CASSANDRA_NODES=$N_NODES
    if [ "$DISJOINT" == "1" ]; then 
        if [ "0$APP_NODES" == "0" ]; then
            echo "ERROR: If the disjoint option is provided a number of application nodes (-nA=<NUM>) should be provided. Aborting..."
            exit
        fi  
        TOTAL_NODES=$((N_NODES + APP_NODES))
    elif [ "0$TOTAL_NODES" == "0" ]; then
        TOTAL_NODES=$N_NODES
    fi

    if [ "0$APP" != "0" ] && [ "0$APP_NODES" == "0" ]; then
        APP_NODES=$N_NODES
    fi

    if [ "0$JOB_MAX_TIME" == "0" ]; then
        JOB_MAX_TIME=$DEFAULT_MAX_TIME
    fi

    echo "[ PARAM DEBUG ]"
    echo "UNIQ_ID: "$UNIQ_ID
    echo "CASSANDRA_NODES: "$CASSANDRA_NODES
    echo "APP_NODES: "$APP_NODES
    echo "PYCOMPSS_SET: "$PYCOMPSS_SET
    echo "DISJOINT: "$DISJOINT
     
    echo sbatch --job-name="$UNIQ_ID" --nodes=$TOTAL_NODES --time=$JOB_MAX_TIME --exclusive --output=$LOGS_DIR/cassandra-%j.out --error=$LOGS_DIR/cassandra-%j.err $QUEUE $CONSTRAINTS $MODULE_PATH/job.sh $UNIQ_ID $CASSANDRA_NODES $APP_NODES $PYCOMPSS_SET $DISJOINT
    SUBMIT_MSG=$(sbatch --job-name="$UNIQ_ID" --nodes=$TOTAL_NODES --time=$JOB_MAX_TIME --exclusive --output=$LOGS_DIR/cassandra-%j.out --error=$LOGS_DIR/cassandra-%j.err $QUEUE $CONSTRAINTS $MODULE_PATH/job.sh $UNIQ_ID $CASSANDRA_NODES $APP_NODES $PYCOMPSS_SET $DISJOINT) 
    echo $SUBMIT_MSG" ("$UNIQ_ID")"
    JOB_NUMBER=$(echo $SUBMIT_MSG | awk '{ print $NF }')
    echo $JOB_NUMBER $UNIQ_ID" " >> $C4S_JOBLIST

    echo "Launching $TOTAL_NODES nodes to recover snapshot $input_snap"
    sleep 5
    echo "Launch still in progress. You can check it later with:"
    echo "    $EXEC_NAME STATUS"
elif [ "$ACTION" == "STOP" ] || [ "$ACTION" == "stop" ]
then
    # If there is a running Cassandra Cluster it stops it
    if [ "0$JOBNAME" == "0" ]; then
        if [ $(cat $C4S_JOBLIST | wc -l) -gt 1 ]; then
            echo "There are many Cassandra clusters in the queuing system, specify which one do you want to stop."
            echo "e.g. $EXEC_NAME STOP c4s1337"
            squeue
        elif [ $(cat $C4S_JOBLIST | wc -l) -eq 1 ]; then
            JOBNAME=$(cat $C4S_JOBLIST | awk '{ print $NF }')
            echo "[INFO] Stopping cluster "$JOBNAME", it may take a while..."
            echo "1" > $C4S_HOME/stop."$JOBNAME".txt
            sleep 5
        else
            exit_no_cluster
        fi
    elif [ $(cat $C4S_JOBLIST | grep " $JOBNAME " | wc -l) -eq 1 ]; then
        # This finishes the running job safely, after making a snapshot if it was created that way.
        echo "[INFO] Stopping cluster "$JOBNAME", it may take a while..."
        echo "1" > $C4S_HOME/stop."$JOBNAME".txt
        sleep 10
    else
        echo "ERROR: Cluster with name "$JOBNAME" not found in the queuing system."
        if [ $(cat $C4S_JOBLIST | wc -l) -gt 0 ]; then
            echo "JOBID    JOBNAME"
            cat $C4S_JOBLIST
            echo "Exiting..."
        else
            echo "There are no clusters in the queueing system. Exiting..."
            exit
        fi
    fi
elif [ "$ACTION" == "KILL" ] || [ "$ACTION" == "kill" ]
then
    # If there is a running Cassandra Cluster it kills it
    if [ "0$JOBNAME" == "0" ]; then
        if [ $(cat $C4S_JOBLIST | wc -l) -gt 1 ]; then
            echo "There are many Cassandra clusters in the queuing system, specify which one do you want to kill."
            echo "e.g. $EXEC_NAME KILL c4s1337"
            squeue
        elif [ $(cat $C4S_JOBLIST | wc -l) -eq 1 ]; then
            JOBNAME=$(cat $C4S_JOBLIST | awk '{ print $NF }')
        fi
    fi
    get_job_info
    if [ "$JOB_ID" != "" ]
    then
        scancel $JOB_ID
        echo "It will take a while to complete the shutdown..." 
        sleep 5
        echo "Done."
    else
        exit_no_cluster
    fi
else
    # There may be an error with the arguments used, also prints the help
    echo "Input argument error. Only an ACTION must be specified."
    usage
    echo "Exiting..."
    exit
fi
