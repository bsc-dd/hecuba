#!/bin/bash

INSTALL_DIR=${1}


mkdir -p ${INSTALL_DIR}/lib/cassandra4slurm

if [ $? -ne 0 ]
then
    echo "Can't create install path ${INSTALL_DIR}/lib/cassandra4slurm" 1>&2
    exit 1
fi

mkdir -p ${INSTALL_DIR}/bin

cp ./scripts/* ${INSTALL_DIR}/lib/cassandra4slurm
ln -s ../lib/cassandra4slurm/launcher.sh  ${INSTALL_DIR}/bin/c4s
ln -s ../lib/cassandra4slurm/execute.sh ${INSTALL_DIR}/bin/runapp
ln -s ../bin/c4s ${INSTALL_DIR}/bin/cassandra4slurm

