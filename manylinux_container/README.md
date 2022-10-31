Hecuba deployment
=================

Hecuba has a C++ core (libhfetch.so) and a Python wrapper to it.

To create the required 'wheel' format compatible with Pypy we need to use one of the manylinux_* docker images from [Pypa](https://github.com/pypa/manylinux).
Then the Python 'build' module is invoked:
'''
    $ python -m build
'''

Which will build the library and the wrapper.

The problem
-----------

The library has dependences:
    - arrow
    - tbb
    - cassandra
    - boost
    - rdkafka
    - uv
    - yaml-cpp

These dependences may be present in the system (and they will be used) or they
will be downloaded and recompiled, finally generating the required wheel.

The problem, is that the 'build' module starts an empty environment without
those dependencies, and therefore it takes a loooong time to generate.

Our solution
------------
In order to reduce the time it takes to download and compile the dependencies,
we opted to build a Docker image that starts from one of the recent 'manylinux'
image and recompiles the dependencies inside. This way, it will take a long
time... but just once. And afterwards, the dependencies are already there and
can be reused.  We use the 'manylinux2014_x86_64':

    $ PLAT='manylinux2014_x86_64'

So, from the directory with the 'Dockerfile':

    $ DOCKER_IMAGE=bsc_dd/hecubacassandra
    $ docker build -t ${DOCKER_IMAGE}

Once the container image is generated, just run the script 'build-wheels.sh' in
the container mapping the source directory ('$pwd' in the example) with a
directory named '/io' inside the container:

    $ docker container run --rm -it -e PLAT=$PLAT  \
        -v "$(pwd)":/io  \
        -e CMAKE_PARALLEL_BUILD=4 \
        "$DOCKER_IMAGE" /io/manylinux/build-wheels.sh

This will generate the 'wheelhouse' directory containing all the required
wheels and you can upload them to pypy.

Hecuba Distribution
-------------------

1) Upgrade the tool to upload the package 'twine':

    $ python3 -m pip install --upgrade twine

2) Ensure the version number has been increased! Increase VERSION.txt and other
   references. And regenerate wheels.
   REMEMBER that once generated and uploaded there is no way back, so in case
   of errors just increase the version.

    $ ... increase version number ...

3) Upload a version to 'testPypy' first!
    This implies that the 'hecuba' name of the package MUST be changed because
    somebody used the name before T_T (We use 'hecuba-bsc-upc' in the meantime)

    $ ... change the package name ...
    $ python3 -m twine upload --repository testpypi wheelhouse/Hecuba*


4) Test the installation.
    (Ideally you should create a virtual environment)

    $ pip install -r requirements.txt
    $ pip install -i https://test.pypi.org/simple/ Hecuba-bsc-upc

5) If everything suits your needs, upload to Pypy for real now.

    $ ... restore the package name ...
    $ python3 -m twine upload wheelhouse/Hecuba*


