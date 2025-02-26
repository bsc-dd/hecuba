#!/bin/bash

DIR="$(dirname $0)" #DIR="${0%/*}" #in case 'dirname' is missing
DIR="$(cd $DIR && pwd)"
INSTALL_DIR=${1}

echo " [INFO] Installing files:"
echo "          From [$DIR]"
echo "          To   [$INSTALL_DIR]"


mkdir -p ${INSTALL_DIR}/bin/cassandra4slurm

if [ $? -ne 0 ]
then
    echo "Can't create install path ${INSTALL_DIR}/bin/cassandra4slurm" 1>&2
    exit 1
fi

mkdir -p ${INSTALL_DIR}/bin/cassandra4slurm/storage_home
mkdir -p ${INSTALL_DIR}/compss/ITF

cp ${DIR}/scripts/* ${INSTALL_DIR}/bin/cassandra4slurm
cp ${DIR}/enqueue_cass_node.sh ${INSTALL_DIR}/bin/cassandra4slurm
cp ${DIR}/storage_home/* ${INSTALL_DIR}/bin/cassandra4slurm/storage_home
if [ -e ${INSTALL_DIR}/compss/scripts ] ; then
    [ ! -L ${INSTALL_DIR}/compss/scripts ] \
        &&  echo "WARNING: Already existing directory [${INSTALL_DIR}/compss/scripts ]"
else
    ln -sf ${INSTALL_DIR}/bin/cassandra4slurm/storage_home ${INSTALL_DIR}/compss/scripts
fi

ln -sf ${INSTALL_DIR}/bin/cassandra4slurm/launcher.sh  ${INSTALL_DIR}/bin/c4s
ln -sf ${INSTALL_DIR}/bin/cassandra4slurm/execute.sh ${INSTALL_DIR}/bin/runapp
cp ${DIR}/../storageAPI/storageItf/target/*jar ${INSTALL_DIR}/compss/ITF

