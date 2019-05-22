#!/bin/bash
set -e -x

# Install a system package required by our library
yum remove -y cmake
yum install -y atlas-devel openssl openssl-devel cmake3  wget
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


# getting TBB
cd;wget https://www.threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb2017_20161128oss_src.tgz
tar -xf tbb2017_20161128oss_src.tgz
cd tbb2017_20161128oss/
make tbb
cp -r build/*_release/libtbb.so*  /usr/lib64
cp -r include/tbb /usr/include
cd

ORIGINAL_CPATH=${CPATH}
ORIGINAL_LD=${LD_LIBRARY_PATH}
export CFLAGS='-std=c++11'
# Compile wheels
for PYBIN in /opt/python/py27*; do
     VNAME=`basename ${PYBIN}`
     export CPATH=$PYBIN/include:${ORIGINAL_CPATH}
     export LD_LIBRARY_PATH=${PYBIN}/lib:/io/build/lib:${ORIGINAL_LD}

    "${PYBIN}/bin/pip" install -r /io/requirements.txt
    cd /io
    rm -rf build dist
    "${PYBIN}/bin/python" setup.py install
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
