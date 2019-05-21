#!/bin/bash
set -e -x

# Install a system package required by our library
yum remove -y cmake
yum install -y atlas-devel openssl openssl-devel cmake3 tbb-devel wget
ln /usr/bin/cmake3 /usr/bin/cmake

# getting the latest libuv
wget https://github.com/libuv/libuv/archive/v1.29.1.tar.gz
tar -xf v1.29.1.tar.gz
cd libuv-1.29.1/
sh autogen.sh
./configure
make install
# getting the latest Cassandra cpp driver
cd; wget https://github.com/datastax/cpp-driver/archive/2.12.0.tar.gz
tar -xf 2.12.0.tar.gz
cd cpp-driver-2.12.0/
cmake -H. -Bbuild
cd build;make;make install


ORIGINAL_CPATH=${CPATH}
ORIGINAL_LD=${LD_LIBRARY_PATH}
export CFLAGS='-std=c++11'
# Compile wheels
for PYBIN in /opt/python/*; do
     VNAME=`basename ${PYBIN}`
     export CPATH=$PYBIN/include:${ORIGINAL_CPATH}
     export LD_LIBRARY_PATH=${PYBIN}/lib:$PWD/build/lib:${ORIGINAL_LD}

    "${PYBIN}/bin/pip" install -r /io/requirements.txt
    "${PYBIN}/bin/pip" wheel /io/ -w wheelhouse/${VNAME}

     # Bundle external shared libraries into the wheels
    for whl in wheelhouse/${VNAME}/*.whl; do
        auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
    done
done
export CPATH=${ORIGINAL_CPATH}
export LD_LIBRARY_PATH=${ORIGINAL_LD}


# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install hecuba --no-index -f /io/wheelhouse/
    (cd "$HOME"; "${PYBIN}/nosetests" hecuba)
done
