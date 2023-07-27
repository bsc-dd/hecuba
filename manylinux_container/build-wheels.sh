#!/bin/bash
#   Create a directory '/io/wheelhouse/' with the  corresponding wheel package
#   for different python versions.  
#   This script must be executed from a container that has '/io' mapped to the
#   hecuba source directory.  
#   The container MUST be based from a 'manylinux_*' image and, for performance
#   reasons, it should have the required dependencies already installed
#   (otherwise they will be downloaded and regenerated).
#
set -e -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}


ORIGINAL_CPATH=${CPATH}
ORIGINAL_LD=${LD_LIBRARY_PATH}
# Compile wheels
for PYBIN in /opt/python/cp3*; do
    VNAME=`basename ${PYBIN}`
    if [ "$VNAME" == "cp312-cp312" ]; then
       continue
    fi
    export CPATH=$PYBIN/include:/io/build/include:/usr/local/include:${ORIGINAL_CPATH}
    # Add libraries path: /io/build/lib #Building directory (just in case) 
    #                     /usr/local/lib #Installed  directory (arrow, boost, rdkafka,uv)
    #                     /usr/local/lib64 #Installed  directory (cassandra, yaml-cpp)
    export LD_LIBRARY_PATH=${PYBIN}/lib:/io/build/lib:/usr/local/lib:/usr/local/lib64:${ORIGINAL_LD}

    "${PYBIN}/bin/pip" install -r /io/requirements.txt
    cd /io
    rm -rf build dist Hecuba.egg-info hecuba_py/hecuba/bin
    "${PYBIN}/bin/python" -m build
    mkdir -p wheelhouse/${VNAME}
    mv dist/* wheelhouse/${VNAME}
    # Bundle external shared libraries into the wheels
    for whl in wheelhouse/${VNAME}/*.whl; do
        repair_wheel "$whl"
    done
done
export CPATH=${ORIGINAL_CPATH}
export LD_LIBRARY_PATH=${ORIGINAL_LD}
