#!/bin/bash

set -x

sed -i /etc/ssh/sshd_config -e 's+PermitRootLogin prohibit-password+PermitRootLogin without-password+'

/usr/sbin/sshd -D &

export PYCOMPSS_NODES=localhost

wget https://raw.githubusercontent.com/bsc-wdc/apps/stable/python/examples/sort/src/sort.py
wget https://raw.githubusercontent.com/bsc-wdc/apps/stable/python/examples/sort/generator/src/generator.py

python3 generator.py 102400 200000 dataset.txt

python3 -c 'import hecuba'

/opt/COMPSs/Runtime/scripts/user/runcompss -d --master_name="localhost" sort.py dataset.txt 5 600

