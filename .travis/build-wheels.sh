#!/bin/bash
set -e -x
export CFLAGS="-std=c11 $CFLAGS"
# Install a system package required by our library
#yum remove -y cmake
yum install -y  cmake  java #openssl openssl-devel
if test -f /usr/bin/cmake3; then
    ln /usr/bin/cmake3 /usr/bin/cmake
else
    curl -L https://github.com/Kitware/CMake/releases/download/v3.3.2/cmake-3.3.2-Linux-x86_64.tar.gz|tar xz
    cd cmake-3.3.2-Linux-x86_64
    cp -r * /usr/
fi

# getting the latest libuv
LIBUV_V=1.19.2  # With versions > 1.19.2 it fails compiling with manylinux1
curl -L https://github.com/libuv/libuv/archive/v${LIBUV_V}.tar.gz|tar -xz
cd libuv-${LIBUV_V}/
sh autogen.sh
./configure
make install
# getting the latest Cassandra cpp driver
cd; curl -L https://github.com/datastax/cpp-driver/archive/2.12.0.tar.gz|tar -xz
cd cpp-driver-2.12.0/
cmake -H. -Bbuild -DCASS_USE_LIBSSH2=OFF -DCASS_USE_OPENSSL=OFF
cd build;make;make install


# getting TBB
cd;curl -L https://github.com/intel/tbb/archive/v2020.0.tar.gz|tar -xz
cd tbb-2020.0/
make tbb
cp -r build/*_release/libtbb.so*  /usr/lib64
cp -r include/tbb /usr/include
cd

ORIGINAL_CPATH=${CPATH}
ORIGINAL_LD=${LD_LIBRARY_PATH}
# Compile wheels
for PYBIN in /opt/python/cp3*; do
     VNAME=`basename ${PYBIN}`
     export CPATH=$PYBIN/include:${ORIGINAL_CPATH}
     export LD_LIBRARY_PATH=${PYBIN}/lib:/io/build/lib:${ORIGINAL_LD}

    "${PYBIN}/bin/pip" install -r /io/requirements.txt
    cd /io
    rm -rf build dist
    "${PYBIN}/bin/python" setup.py build
    "${PYBIN}/bin/pip" wheel /io/ -w wheelhouse/${VNAME}

     # Bundle external shared libraries into the wheels
    for whl in wheelhouse/${VNAME}/*.whl; do
        auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
    done
done
export CPATH=${ORIGINAL_CPATH}
export LD_LIBRARY_PATH=${ORIGINAL_LD}


# Install packages and test
#for PYBIN in /opt/python/cp3*/bin/; do
#    "${PYBIN}/pip" install hecuba --no-index -f /io/wheelhouse/
#    "${PYBIN}/nosetests" --with-coverage -v -s /io/hecuba_py/tests/*.py
#    "${PYBIN}/nosetests" --with-coverage -v -s /io/hecuba_py/tests/withcassandra
#done

