#!/bin/bash
#module load boost/1.64.0_py2 COMPSs/2.4.rc1903
#module load COMPSs/2.4.rc1905
if [ "${1}" != "-" ]; then
    interface=${1} # Receiving the interface, if any. (i.e. -ib0)
fi

export CONTACT_NAMES=$(cat PLACEHOLDER_CASSANDRA_NODES_FILE | paste -s -d ' ' | sed "s/ /$interface,/g")$interface
PYCOMPSS_NODES_FILE=PLACEHOLDER_PYCOMPSS_NODES_FILE
echo "PYCOMPSS_NODES_FILE="$PYCOMPSS_NODES_FILE
PYCOMPSS_MASTER=$(head -n 1 $PYCOMPSS_NODES_FILE | sed "s/$interface//")
echo "PYCOMPSS_MASTER is: "$PYCOMPSS_MASTER
PYCOMPSS_WORKERS=$(tail -n +2 $PYCOMPSS_NODES_FILE | paste -s -d ' ' | sed "s/$interface//g")
echo "PYCOMPSS_WORKERS are: "$PYCOMPSS_WORKERS
cmd=(launch_compss --master_node=$PYCOMPSS_MASTER --worker_nodes="$PYCOMPSS_WORKERS" --classpath=/home/bsc31/bsc31906/hecuba-src/storageLtf/StorageItf-1.0-jar-with-dependencies.jar --storage_conf=PLACEHOLDER_PYCOMPSS_STORAGE PLACEHOLDER_PYCOMPSS_FLAGS PLACEHOLDER_APP_PATH_AND_PARAMETERS)
"${cmd[@]}"
# Placeholder:look-a-like
# PLACEHOLDER_CASSANDRA_NODES_FILE:$HOME/bla.txt
# PLACEHOLDER_PYCOMPSS_NODES_FILE:pycompssnodes.txt
# PLACEHOLDER_APP_PATH_AND_PARAMETERS:$HOME/4_init_parallel_PSCOs/src/matmul.py 4 32 4 16
# PLACEHOLDER_PYCOMPSS_FLAGS:-d --pythonpath=/path/to/somewhere/ --lang=python
