#!/bin/bash

echo "===== Initialize Hecuba ===="
echo " CQLSH at ${1}"
HOSTNAME=${1}
shift
# CQLSH_HOST and CQLSH_PORT are environment variables that can be set to connect to a host different than local host

#if [ "x$CQLSH_HOST" == "x" ] ; then
#	export CQLSH_HOST=localhost
#fi
CQLSH_CMD="${@}" #second parameter cqlsh location. To support execution with singularity (currently only supported for pycompss applications launched by enqueue_compss)
echo " CQLSH_CMD [$CQLSH_CMD]"

# CQLSH 6.0x does NOT accept stdout or '-e' flag, flag '-f' works for version 5.0 and forward
# Create a temporal file to store CQL commands
TMP=$(mktemp -t hecuba.tmp.XXXXXX)

trap "cleanup $TMP" SIGINT SIGQUIT SIGABRT

cleanup() {
    local TMP="$1"
    echo " Removing temporal file [$TMP]"
    rm -rf $TMP
}

commands="CREATE KEYSPACE IF NOT EXISTS hecuba_locks WITH replication=  {'class': 'SimpleStrategy', 'replication_factor': 1}; CREATE TABLE IF NOT EXISTS hecuba_locks.table_lock (table_name text, PRIMARY KEY (table_name)); TRUNCATE table hecuba_locks.table_lock;"

echo $commands > $TMP
eval "${CQLSH_CMD} ${HOSTNAME} -f $TMP" || { echo "[ERROR]: Executing CQLSH [$commands] FAILED!!!" && exit; }

echo "  Keyspace hecuba_locks created"
echo "  Table    table_lock created"


commands="CREATE KEYSPACE IF NOT EXISTS hecuba WITH replication=  {'class': 'SimpleStrategy', 'replication_factor': 1}; CREATE TYPE IF NOT EXISTS hecuba.q_meta( mem_filter text, from_point frozen<list<double>>, to_point frozen<list<double>>, precision float); CREATE TYPE IF NOT EXISTS hecuba.np_meta (flags int, elem_size int, partition_type tinyint, dims list<int>, strides list<int>, typekind text, byteorder text); CREATE TABLE IF NOT EXISTS hecuba.istorage (storage_id uuid, class_name text,name text, istorage_props map<text,text>, tokens list<frozen<tuple<bigint,bigint>>>, indexed_on list<text>, qbeast_random text, qbeast_meta frozen<q_meta>, numpy_meta frozen<np_meta>, saved_numpy_meta frozen<np_meta>, block_id int, base_numpy uuid, view_serialization blob, primary_keys list<frozen<tuple<text,text>>>, columns list<frozen<tuple<text,text>>>, PRIMARY KEY(storage_id));"


echo $commands > $TMP
eval "${CQLSH_CMD} ${HOSTNAME} -f $TMP" || { echo "[ERROR]: Executing CQLSH [$commands] FAILED!!!" && exit; }

echo "  Keyspace hecuba created"
echo "  Table    istorage created"

[ "0${EXECUTION_NAME}" == "0" ] && EXECUTION_NAME="my_app"
[ "0${REPLICATION_STRATEGY}" == "0" ] && REPLICATION_STRATEGY="SimpleStrategy"
[ "0${REPLICA_FACTOR}" == "0" ] && REPLICA_FACTOR="1"
[ "0${REPLICATION_STRATEGY_OPTIONS}" == "0" ] && REPLICATION_STRATEGY_OPTIONS="''"

if [ ${REPLICATION_STRATEGY} == "SimpleStrategy" ]; then
commands="CREATE KEYSPACE IF NOT EXISTS ${EXECUTION_NAME} WITH replication=  {'class': '${REPLICATION_STRATEGY}', 'replication_factor': ${REPLICA_FACTOR}}; "
else
commands="CREATE KEYSPACE IF NOT EXISTS ${EXECUTION_NAME} WITH replication=  {'class': '${REPLICATION_STRATEGY}', ${REPLICATION_STRATEGY_OPTIONS}}; "
fi


echo $commands > $TMP
eval "${CQLSH_CMD} ${HOSTNAME} -f $TMP" || { echo "[ERROR]: Executing CQLSH [$commands] FAILED!!!" && exit; }

echo "  Keyspace ${EXECUTION_NAME} created"

[ "0${HECUBA_SN_SINGLE_TABLE}" == "0" ] && HECUBA_SN_SINGLE_TABLE="true"
if [[ ${HECUBA_SN_SINGLE_TABLE} != "no" && ${HECUBA_SN_SINGLE_TABLE} != "false" ]]; then
    commands="CREATE TABLE IF NOT EXISTS ${EXECUTION_NAME}.hecuba_storagenumpy (storage_id uuid, cluster_id int, block_id int, payload blob, PRIMARY KEY ((storage_id, cluster_id), block_id));"
    echo $commands > $TMP
    eval "${CQLSH_CMD} ${HOSTNAME} -f $TMP" || { echo "[ERROR]: Executing CQLSH [$commands] FAILED!!!" && exit; }
    echo "  Table    hecuba_storagenumpy created"
fi

echo "Hecuba initialization completed"

export CREATE_SCHEMA=False

cleanup $TMP
