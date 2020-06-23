#!/bin/bash

echo "===== Initialize Hecuba ===="
# CQLSH_HOST and CQLSH_PORT are environment variables that can be set to connect to a host different than local host

#if [ "x$CQLSH_HOST" == "x" ] ; then
#	export CQLSH_HOST=localhost
#fi

commands="CREATE KEYSPACE IF NOT EXISTS hecuba_locks WITH replication=  {'class': 'SimpleStrategy', 'replication_factor': 1}; CREATE TABLE IF NOT EXISTS hecuba_locks.table_lock (table_name text, PRIMARY KEY (table_name)); TRUNCATE table hecuba_locks.table_lock;"

# Change to Python 2.7 to execute CQLSH
module load python/2.7.16  
echo $commands | cqlsh ${1}

echo " CQLSH at ${1}"
echo "  Keyspace hecuba_locks created"
echo "  Table    table_lock created"


commands="CREATE KEYSPACE IF NOT EXISTS hecuba WITH replication=  {'class': 'SimpleStrategy', 'replication_factor': 1}; CREATE TYPE IF NOT EXISTS hecuba.q_meta( mem_filter text, from_point frozen<list<double>>, to_point frozen<list<double>>, precision float); CREATE TYPE IF NOT EXISTS hecuba.np_meta (flags int, elem_size int, partition_type tinyint, dims list<int>, strides list<int>, typekind text, byteorder text); CREATE TABLE IF NOT EXISTS hecuba.istorage (storage_id uuid, class_name text,name text, istorage_props map<text,text>, tokens list<frozen<tuple<bigint,bigint>>>, indexed_on list<text>, qbeast_random text, qbeast_meta frozen<q_meta>, numpy_meta frozen<np_meta>, saved_numpy_meta frozen<np_meta>, block_id int, base_numpy uuid, primary_keys list<frozen<tuple<text,text>>>, columns list<frozen<tuple<text,text>>>, PRIMARY KEY(storage_id));"


echo $commands | cqlsh ${1}

echo "  Keyspace hecuba created"
echo "  Table    istorage created"

echo "Hecuba initialization completed"

export CREATE_SCHEMA=1

# Restore Python 3 version
module load python/3.6.4_ML  
