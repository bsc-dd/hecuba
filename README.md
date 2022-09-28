# Hecuba ![](https://travis-ci.org/bsc-dd/hecuba.svg?branch=master) [![codecov](https://codecov.io/gh/bsc-dd/hecuba/branch/master/graph/badge.svg)](https://codecov.io/gh/bsc-dd/hecuba)
Non-relational databases are nowadays a common solution when dealing with a huge data set and massive query workload. These systems have been redesigned from scratch to achieve scalability and availability at the cost of providing only a reduced set of low-level functionality, thus forcing the client application to take care of complex logic. As a solution, our research group developed **Hecuba**, a set of tools and interfaces, which aims to facilitate programmers with an efficient and easy interaction with non-relational technologies.

## Installation procedure

### Software requisites:

+ GCC >= 5.4.0
+ CMake >= 3.14
+ Python >= 3.6 development version installed.

### Dependencies:
+ Cassandra >= 4.0 [Github](https://github.com/apache/cassandra)
+ Kafka >= 2.13

### Auto-downloaded dependencies (during the installation process):
#### python dependencies
+ numpy library >= 1.16
+ Cassandra driver for python >= 3.7.1
+ nose >= 1.3.7
+ ccm
+ mock
#### c++ dependencies
+ Cassandra driver for C++ >= 2.14.1 [Github](https://github.com/datastax/cpp-driver)
+ libTBB >= 2020.0 (Intel Threading Building Blocks) [Github](https://github.com/01org/tbb)
+ libuv >= 1.11.0 [Github](https://github.com/libuv/libuv)
+ Apache Arrow >= 0.15.1 [Github](https://github.com/apache/arrow)
+ yaml-cpp >= 0.7.0 [Github](https://github.com/jbeder/yaml-cpp)
+ librdkafka >= 1.9.2 [Github](https://github.com/edenhill/librdkafka)

<!--- ### Quick install in OpenSuse 42.2

```bash
# Install dependencies
sudo zypper install cmake python-devel gcc-c++ python-numpy-devel python-pip

pip install hecuba
```
-->

## Manual installation

This procedure launches CMake to build the dependencies, which can take some time.

If CMake selects the wrong compiler, it can be replaced by defining the environment vars `CC=/custom/path/gcc` and `CXX=/custom/path/g++`. Then remove the build folder and restart the install process.
 
If the compilation fails because some feature is not supported or requires C++11 support, define the environment variable `CFLAGS="-std=c++11 $CFLAGS" and resume the installation.


The first step is to download the code.

```bash
# Clone the repository
git clone https://github.com/bsc-dd/hecuba.git hecuba_repo
cd hecuba_repo
```

Then it is only necessary to run the `setup.py` python script, which performs all the steps to compile and install Hecuba in the system. Notice that Hecuba is composed by Python code and C++ code. The `setup.py` script, on the one hand, compiles the C++ code and installs the C++ header files and libraries, and on the other hand, generates and installs a Python package.
You need to decide where to install Hecuba:

```bash
# (1) To install hecuba to the default system directory
python setup.py install
# (2) Install to user space, under $HOME/.local
python setup.py install --user
# (3) Install to a user-defined path $CUSTOM_PATH
python setup.py install --prefix=$CUSTOM_PATH
# (4) Install to a user-defined path $CUSTOM_PATH and with a custom directory for the libraries
python setup.py install --prefix=$CUSTOM_PATH --c_binding=$HECUBA_LIBS_PATH
```

The option `--c_binding` indicates the target location for the C++ libraries (`$HECUBA_LIBS_PATH/lib`) and headers (`$HECUBA_LIBS_PATH/include`). If this option is not specified then the target directory will be the same than the target directory for the Python package.

The target directory for the python package can be the default system directory (1), the user space (`$HOME/.local`) (2) or a custom path (3 and 4).

Warning: Be sure that the `PYTHONPATH` variable contains the path to the Hecuba Python package and that the `LD_LIBRARY_PATH` contains the path of the C++ Hecuba libraries.


### Auto-downloading process

Before starting the compilation, the installation procedure checks if the required libraries are already in the directories specified as the compilation directories or accessible via `LD_LIBRARY_PATH`. If they are not there, then it checks if their source code is in the directory `hecuba_core/dependencies`. If it is not there, then it downloads its source code.

Note: If you want to install Hecuba on a computer without internet access, first make an initial installation on a machine that has internet access and then copy the files under `hecuba_core/dependencies` to the remote computer under the same directory.



### Install the Hecuba core only

If you need just the C++ interface of Hecuba and want to skip the Python installation you can just build the C++ side using the following comands:

```bash
cmake -Hhecuba_core -Bbuild -DC_BINDING_INSTALL_PREFIX=$HECUBA_LIBS_PATH
make -C build
```

This will install under the `HECUBA_LIBS_PATH/lib` folder the C++ libraries and under `HECUBA_LIBS_PATH/include` the headers.


## Instructions to execute with Hecuba:

Please, refer to the [Hecuba manual](https://github.com/bsc-dd/hecuba/wiki/1:-User-Manual) for the execution instructions.


## LICENSING 

Copyright 2017 Barcelona Supercomputing Center

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
