#!/bin/bash

set -x

sed -i /etc/ssh/sshd_config -e 's+PermitRootLogin prohibit-password+PermitRootLogin without-password+'

/usr/sbin/sshd -D &

export PYCOMPSS_NODES=localhost

# COMPSs Test
#wget https://raw.githubusercontent.com/bsc-wdc/apps/stable/python/examples/sort/src/sort.py
#wget https://raw.githubusercontent.com/bsc-wdc/apps/stable/python/examples/sort/generator/src/generator.py

#python3 generator.py 102400 200000 dataset.txt

#/opt/COMPSs/Runtime/scripts/user/runcompss -d --master_name="localhost" sort.py dataset.txt 5 600


# Install current version of Hecuba
pip3 uninstall -y hecuba

cd /io
python3 setup.py install
cd

cd /io/storageAPI/storageItf
mvn clean package
#export CLASSPATH=/io/storageAPI/storageItf/target/StorageItf-1.0.jar:$CLASSPATH
cd

# Cassandra is UP?
touch /tmp/nodedown_err

retries=0
max_retry=10
until python3 /io/compss_integration_tests/scripts/cassandra_is_up.py $CONTACT_NAMES 2> /tmp/nodedown_err
do
  sleep 5
  if [ $retries -eq $max_retry ]
  then
    cat /tmp/nodedown_err
    exit 1
  fi
  retries=$(( $retries + 1 ))
done

# Run integration test
python3 -c 'import hecuba'
echo "export CONTACT_NAMES=${CONTACT_NAMES}" >> ${HOME}/.bashrc

for test in /io/compss_integration_tests/*.py
do
  /opt/COMPSs/Runtime/scripts/user/runcompss -d --python_interpreter=python3 --master_name="localhost" --classpath=/io/storageAPI/storageItf/target/StorageItf-1.0.jar  $test
  if [ $? -ne 0 ]
  then
    echo "Test $test failed"
    exit 1
  fi
done

exit 0

