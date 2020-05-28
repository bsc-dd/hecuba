#!/bin/bash
###############################################################################################################
#                                                                                                             #
#                                        Application Launcher for Slurm                                       #
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
# Cleaning old executions (more than a month ago)
`find $C4S_HOME -maxdepth 1 -mtime +30 -type f | grep -v ".cfg" | xargs rm -f`
# Cleaning old jobs (not in the queuing system anymore)
C4S_JOBLIST=$C4S_HOME/joblist.txt
C4S_SCONTROL=$C4S_HOME/scontrol.txt
C4S_SQUEUE=$C4S_HOME/squeue.txt
APP_JOBLIST=$C4S_HOME/appjoblist.txt
scontrol show job > $C4S_SCONTROL
squeue > $C4S_SQUEUE
touch $C4S_JOBLIST $C4S_HOME/newjoblist.txt $C4S_HOME/newappjoblist.txt
if [ $(squeue | wc -l) -eq 1 ]; then
    rm -f $C4S_JOBLIST $APP_JOBLIST; touch $C4S_JOBLIST $APP_JOBLIST
else
    OLDIFS=$IFS; IFS=$'\n';
    for job_line in $(cat $C4S_JOBLIST); do
        job_id=$(echo $job_line | awk '{ print $1 }')
        if [ $(grep "$job_id " $C4S_SQUEUE | wc -l) -gt 0 ]; then
            echo "$job_line" >> $C4S_HOME/newjoblist.txt
        fi
    done
    for job_line in $(cat $APP_JOBLIST); do
        job_id=$(echo $job_line | awk '{ print $1 }')
        if [ $(grep "$job_id " $C4S_SQUEUE | wc -l) -gt 0 ]; then
            echo "$job_line" >> $C4S_HOME/newappjoblist.txt
        fi
    done
    IFS=$OLDIFS
    mv $C4S_HOME/newjoblist.txt $C4S_JOBLIST
    mv $C4S_HOME/newappjoblist.txt $APP_JOBLIST
fi

CFG_FILE=$C4S_HOME/conf/cassandra4slurm.cfg
CASS_HOME=$(cat $CFG_FILE | grep -v "#" | grep "CASS_HOME=" | tail -n 1 | sed 's/CASS_HOME=//g' | sed 's/"//g' | sed "s/'//g")
MODULE_PATH=$HECUBA_ROOT/bin/cassandra4slurm
UNIQ_ID="c4app"$(echo $RANDOM | cut -c -3)
DEFAULT_APP_NODES=2
DEFAULT_MAX_TIME="00:30:00"
RETRY_MAX=15
PYCOMPSS_SET=0

function usage () {
    # Prints a help message
    echo "Usage: ${0} [ -h | -l | RUN [-n=N_APP_NODES] [-c=cluster_id] --appl=PATH [ARGS] [ --pycompss[=PARAMS] ] [ -t=HH:MM:ss ] [ --qos=debug ] [ --jobname=NAME ] [ --logs=DIR ] | KILL [-e=application_id] ]"
    echo " "
    echo "       -h:"
    echo "       Prints this usage help."
    echo " "
    echo "       -l | --list:"
    echo "       Shows a list of Cassandra clusters and applications."
    echo " "
    echo "       RUN:"
    echo "       Starts a new application execution over an existing Cassandra cluster, depending of the optional parameters."
    echo "       The flag -n is used to set the number of application nodes to reserver. Default is 2."
    echo "       It is used --appl to set the path to the executable and its arguments, if any."
    echo "       If the application must be executed using PyCOMPSs the variable --pycompss should contain its PyCOMPSs parameters."
    echo "       Using -t it will set the maximum time of the job to this value, with HH:MM:ss format. Default is 30 minutes (00:30:00)."
    echo "       Using --qos=debug it will run in the testing queue. It has some restrictions (a single job and 2h max.) so any higher requirements will be rejected by the queuing system."
    echo " "
    echo "       KILL application_id:"
    echo "       The application identified by application_id is killed, aborting the process."
    echo " "
}

function set_utils_paths () {
    export JOBNAME=$UNIQ_ID
    APP_PATH_FILE=$C4S_HOME/app-"$UNIQ_ID".txt
    PYCOMPSS_FLAGS_FILE=$C4S_HOME/pycompss-flags-"$UNIQ_ID".txt
}

function show_list_info () {
    # Gets clusters and application information from SLURM
    num_clusters=$(cat $C4S_JOBLIST | wc -l)
    plural1="This is"
    plural2=""
    if [ $num_clusters -gt 1 ]; then
        plural1="These are"
        plural2="s"
    fi
    if [ "$num_clusters" != "0" ]; then
        echo "$plural1 the existing Cassandra cluster$plural2:"
        echo "JOBID    JOB NAME"
        echo $(cat $C4S_JOBLIST)
    else
        echo "There are no Cassandra clusters."
    fi
    num_apps=$(cat $APP_JOBLIST | wc -l)
    plural3="This is"
    plural4=""
    if [ $num_apps -gt 1 ]; then
        plural3="These are"
        plural4="s"
    fi
    if [ "$num_apps" != "0" ]; then
        echo "$plural3 the existing application$plural4:"
        echo "JOBID    JOB NAME"
        echo $(cat $APP_JOBLIST)
    else
        echo "There are no applications."
    fi
}

function get_job_info () {
    # Gets the ID of the job that runs the Cassandra Cluster
    JOB_INFO=$(squeue | grep c4s) 
    JOB_ID=$(echo $JOB_INFO | awk '{ print $1 }')
    JOB_STATUS=$(echo $JOB_INFO | awk '{ print $5 }')   
}

function get_cluster_node () {
    # Gets the ID of the first node
    #NODE_ID=$(head -n 1 $C4S_HOME/hostlist-$(squeue | grep $JOBNAME | awk '{ print $1 }').txt)
    NODE_ID=$(head -n 1 $C4S_HOME/hostlist-"$(squeue | grep c4s | awk '{ print $3 }')".txt)
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

function test_if_cluster_up () { # Perhaps this function doesnt make sense anymore
    # Checks if there are Cassandra Clusters running, aborting if not
    if [ $(cat $C4S_JOBLIST | wc -l) -eq 0 ]; then
        exit_no_cluster
    elif [ $(cat $C4S_JOBLIST | wc -l) -eq 1 ]; then
        CLUSTID=$(cat $C4S_JOBLIST | awk '{ print $2 }')
        CLUSTST=$(squeue | grep "$(cat $C4S_JOBLIST | awk '{ print $1 }') " | awk '{ print $5 }')
        if [ "$CLUSTST" != "R" ]; then
            echo "ERROR: The job status is not running (R). Exiting..."
            squeue
            exit
        fi
        if [ "0$CLUSTERID" != "0" ] && [ "$CLUSTERID" != "$CLUSTID" ]; then
            echo "ERROR: Given Cluster ID ("$CLUSTERID") not found."
            echo "The only available Cluster is "$CLUSTID". Exiting..."
            exit
        else
            CLUSTERID=$CLUSTID
        fi
    elif [ "0$CLUSTERID" == "0" ]; then
        echo "ERROR: There are many Cassandra clusters, use -c=cluster_name to specify which one to use."
        echo "JOBID   CLUSTER NAME"
        cat $C4S_JOBLIST
        exit         
    elif [ "$(cat $C4S_JOBLIST | grep " $CLUSTERID ")" != $CLUSTERID ]; then
        echo "ERROR: Given Cluster ID ("$CLUSTERID") not found. The available ones are the following:"
        echo "JOBID   CLUSTER NAME"
        cat $C4S_JOBLIST
        exit
    fi
}

function get_nodes_up () {
    get_job_info
    if [ "$JOB_ID" != "" ]
    then
        if [ "$JOB_STATUS" == "R" ]
        then    
            get_cluster_node 
            NODE_STATE_LIST=`ssh -q $NODE_ID "$CASS_HOME/bin/nodetool -h $NODE_ID$CASS_IFACE status" | sed 1,5d | sed '$ d' | awk '{ print $1 }'`
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

for i in "$@"; do
case $i in
    -h|--help)
    usage
    exit
    ;;
    run|RUN)
    ACTION="RUN"
    echo "Action is RUN."
    shift
    ;;
    -l|--list)
    show_list_info
    exit
    shift
    ;;
    kill|KILL)
    ACTION="KILL"
    shift
    ;;
    -a=*|--appl=*)
    APP="${i#*=}"
    echo $APP > $APP_PATH_FILE
    shift
    ;;
    -p=*|--pycompss=*)
    PYCOMPSS_APP="${i#*=}"
    PYCOMPSS_SET=1
    echo $PYCOMPSS_APP > $PYCOMPSS_FLAGS_FILE
    shift
    ;;
    -c=*|--cluster=*)
    CLUSTERID="${i#*=}"
    shift
    ;;
    -t=*|--time=*)
    JOB_MAX_TIME="${i#*=}"
    shift
    ;;
    -n=*|--number_of_nodes=*)
    APP_NODES="${i#*=}"
    shift
    ;;
    -q=*|--qos=*)
    QUEUE="--qos=""${i#*=}"
    shift
    ;;
    -l=*|--logs=*)
    LOGS_DIR="${i#*=}"
    mkdir -p $LOGS_DIR
    shift
    ;;
    -j=*|--jobname=*)
    UNIQ_ID="${i#*=}"
    UNIQ_ID=$(echo $UNIQ_ID | sed 's+ ++g')
    if [ $(grep " $UNIQ_ID " $C4S_HOME/joblist.txt | wc -l) -gt 0 ]; then
        echo "Jobname "$UNIQ_ID" already in use. Continue? (y/n) "
        read input_jobname
        while [ "$input_jobname" != "y" ] && [ "$input_jobname" != "n" ]; do
            echo "Wrong option. Continue? (y/n) "
            read input_jobname
        done
        if [ "$input_jobname" == "n" ]; then
            echo "Aborted."
            exit
        fi
    fi
    shift
    ;;
    -e=*|--execution=*)
    EXEC_ID="${i#*=}"
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
    echo "Check help: ./${0} -h"
    exit
fi

if [ "$ACTION" == "RUN" ]; then
    if [ "0$APP" == "0" ]; then
        echo "ERROR: An application must be specified. Use mandatory flag --appl, or check help for more info. Exiting..."
        exit
    fi
    set_utils_paths
    test_if_cluster_up
    if [ "0$NUM_NODES" == "0" ]; then
        APP_NODES=$DEFAULT_APP_NODES
    fi
    # Check PyCOMPSs pre-condition (at least two nodes, 1 master 1 slave)
    if [ "0$PYCOMPSS_APP" != "0" ] && [ $APP_NODES -lt 2 ]; then
        echo "ERROR: PyCOMPSs executions need at least 2 application nodes. Aborting..."
        exit
    elif [ "0$PYCOMPSS_APP" != "0" ]; then
        echo "[INFO] This execution will use PyCOMPSs in $APP_NODES nodes."
    fi
 
    #WHILE DEBUGGING...
    echo "EXECUTION SUMMARY:"
    #echo "# of Cassandra nodes: "$CASSANDRA_NODES
    echo "# of application nodes: "$APP_NODES
    #echo "# total of requested nodes: "$TOTAL_NODES
    #DEBUGGING#exit

    echo "Job allocation started..."

    
    if [ "0$JOB_MAX_TIME" == "0" ]; then
        JOB_MAX_TIME=$DEFAULT_MAX_TIME
    fi
    if [ "0$LOGS_DIR" == "0" ]; then
        DEFAULT_LOGS_DIR=$(cat $CFG_FILE | grep "LOG_PATH=" | sed 's/LOG_PATH=//g' | sed 's/"//g')
        LOGS_DIR=$DEFAULT_LOGS_DIR
    fi

    sbatch --job-name=$UNIQ_ID --ntasks=$APP_NODES --ntasks-per-node=1 --time=$JOB_MAX_TIME --exclusive $QUEUE --output=$LOGS_DIR/app-%j.out --error=$LOGS_DIR/app-%j.err $MODULE_PATH/job-app.sh $UNIQ_ID $PYCOMPSS_SET $CLUSTER_ID
    sleep 3
    squeue

elif [ "$ACTION" == "KILL" ] || [ "$ACTION" == "kill" ]; then
    # If there is an application it kills it
    if [ $(cat $APP_JOBLIST | wc -l) -eq 0 ]; then
        echo "ERROR: There is no running application to kill. Exiting..."
    elif [ "0$EXEC_ID" != "0" ]; then
        if [ "0$(cat $APP_JOBLIST | grep $EXEC_ID )" == "0" ]; then
            echo "[ERROR] Application name $EXEC_ID not found, these are the current jobs:"
            cat $APP_JOBLIST
        else
            echo "[INFO] Killing application $EXEC_ID. It may take a while..."
            scancel $(cat $APP_JOBLIST | grep " $EXEC_ID " | awk '{ print $1 }')
            echo "Done."
        fi
    elif [ $(cat $APP_JOBLIST | wc -l) -eq 1 ]; then
        JOBID=$(cat $APP_JOBLIST | awk '{ print $1 }')
        JOBNAME=$(cat $APP_JOBLIST | awk '{ print $2 }')
        echo "INFO: Killing application $JOBNAME. It may take a while..."
        scancel $JOBID
        echo "Done."
    fi
    exit
else
    # There may be an error with the arguments used, also prints the help
    echo "Input argument error. Only an ACTION must be specified."
    usage
    echo "Exiting..."
    exit
fi
