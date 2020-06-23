#!/bin/bash

INSTALL_DIR=${1}


mkdir -p ${INSTALL_DIR}/bin/cassandra4slurm

if [ $? -ne 0 ]
then
    echo "Can't create install path ${INSTALL_DIR}/bin/cassandra4slurm" 1>&2
    exit 1
fi

mkdir -p ${INSTALL_DIR}/bin/cassandra4slurm/storage_home
mkdir -p ${INSTALL_DIR}/compss/

cp ./scripts/* ${INSTALL_DIR}/bin/cassandra4slurm
cp ./enqueue_cass_node.sh ${INSTALL_DIR}/bin/cassandra4slurm
cp ./storage_home/* ${INSTALL_DIR}/bin/cassandra4slurm/storage_home
ln -sf ${INSTALL_DIR}/bin/cassandra4slurm/storage_home ${INSTALL_DIR}/compss/scripts
ln -sf ${INSTALL_DIR}/bin/cassandra4slurm/launcher.sh  ${INSTALL_DIR}/bin/c4s
ln -sf ${INSTALL_DIR}/bin/cassandra4slurm/execute.sh ${INSTALL_DIR}/bin/runapp

