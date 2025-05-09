
source $HECUBA_ROOT/bin/cassandra4slurm/hecuba_debug.sh

# update_kafka_configuration
#   Args:   zknode    FQN for the Zookeeper Node
# Update the generic kafka configuration files to the current execution
# environment, in particular, the zookeeper node to use and the directories to
# use.

update_kafka_configuration() {
    local ZKNODE="$1"

    DBG " Kafka configuration: Zookeeper node at [$ZKNODE]"
    DBG " Kafka configuration: Zookeeper logs at [${C4S_HOME}/${UNIQ_ID}/zookeeper]"
    DBG " Kafka configuration: Kafka Logs at [${C4S_HOME}/${UNIQ_ID}/kafka-logs]"

    mkdir -p ${C4S_HOME}/${UNIQ_ID}

    # copy configuration files to EXECUTION directory
    cat ${KAFKA_PATH}/config/server.properties \
        | sed "s/zookeeper.connect=localhost:/zookeeper.connect=${ZKNODE}:/" \
        | sed 's/broker.id=.$/broker.id.generation.enable=true/' \
        > ${C4S_HOME}/${UNIQ_ID}/server.properties

    cat  ${KAFKA_PATH}/config/zookeeper.properties \
        | sed "s#dataDir=/tmp/zookeeper#dataDir=/tmp/${UNIQ_ID}/zookeeper#" \
        > ${C4S_HOME}/${UNIQ_ID}/zookeeper.properties
    DBG " Kafka configuration UPDATED"

}

# Get the directory path for the kafka tools or die
get_kafka_path () {
    local X=$(which kafka-server-start.sh)
    if [ "$?" != "0" ]; then
        die "[ERROR] KAFKA tools NOT FOUND."
    fi
    ## Binary exists, kafka is installed in directory X/..
    echo $(dirname $(dirname "$X"))
}

launch_kafka () {
    local CASSFILE="$1"
    local UNIQ_ID="$2"
    local N_NODES="$3"

    local ZKNODE=$( head -n 1 $CASSFILE )
    local ZKNODEIP=$( head -n 1 $CASSFILE.ips )
    casslist=`cat $CASSFILE`
    local CASSANDRA_NODELIST=$(echo $casslist | sed -e 's+ +,+g')

    KAFKA_PATH=$(get_kafka_path)

    # Prepare configuration files
    update_kafka_configuration "$ZKNODEIP"

    [ -z "$LOG_PATH" ] \
    && echo "ERROR: LOG_PATH not defined. Using default" \
    && export LOG_PATH=$HOME/.c4s/logs

    export LOG_DIR=${LOG_PATH}/${UNIQ_ID}/kafka-logs

    mkdir -p $LOG_DIR

    # Start Zookeeper
    run srun  --overlap --mem=0 --nodelist $ZKNODE --ntasks=1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 \
        --output ${C4S_HOME}/${UNIQ_ID}/zookeeper.output \
         zookeeper-server-start.sh ${C4S_HOME}/${UNIQ_ID}/zookeeper.properties &

    sleep 2


    # Start Kakfa daemons
    local OLDCLASSPATH="$CLASSPATH"
    unset CLASSPATH
    run srun --overlap --mem=0 --nodelist=$CASSANDRA_NODELIST --ntasks=$N_NODES --nodes=$N_NODES --ntasks-per-node=1 --cpus-per-task=1 \
        --output ${C4S_HOME}/${UNIQ_ID}/kafka.output \
        kafka-server-start.sh ${C4S_HOME}/${UNIQ_ID}/server.properties &
    CLASSPATH="$OLDCLASSPATH"

    sleep 2
}
