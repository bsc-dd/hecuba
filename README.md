# Hecuba ![](https://travis-ci.org/bsc-dd/hecuba.svg?branch=master) [![codecov](https://codecov.io/gh/bsc-dd/hecuba/branch/master/graph/badge.svg)](https://codecov.io/gh/bsc-dd/hecuba)
Non-relational databases are nowadays a common solution when dealing with a huge data set and massive query workload. These systems have been redesigned from scratch to achieve scalability and availability at the cost of providing only a reduced set of low-level functionality, thus forcing the client application to take care of complex logic. As a solution, our research group developed **Hecuba**, a set of tools and interfaces, which aims to facilitate programmers with an efficient and easy interaction with non-relational technologies.

## Installation procedure

### Software requisites:

+ GCC >= 5.4.0
+ CMake >= 3.14
+ Python >= 3.6 development version installed.


<!--- ### Quick install in OpenSuse 42.2

```bash
# Install dependencies
sudo zypper install cmake python-devel gcc-c++ python-numpy-devel python-pip

pip install hecuba
```
-->

## Instructions to execute with Hecuba:

1) Start Cassandra.
```bash
$CASSANDRA_HOME/bin/cassandra -f 
```

2) Configure the environment variables as described in [Wiki Launch Hecuba](https://github.com/bsc-dd/hecuba/wiki/1:-User-Manual#how-to-execute).
```bash
# E.g:
export CONTACT_NAMES=127.0.0.1
```

3) Launch the desired application
```bash
python myapp.py
```

Please, refer to the [Hecuba manual](https://github.com/bsc-dd/hecuba/wiki/1:-User-Manual) to check more Hecuba configuration options.


## Manual installation

This procedure launches CMake to build the dependencies, which can take some time.

If CMake selects the wrong compiler, it can be replaced by defining the environment vars `CC=/custom/path/gcc` and `CXX=/custom/path/g++`. Then remove the build folder and restart the install process.
 
If the compilation fails because some feature is not supported or requires C++11 support, define the environment variable `CFLAGS="-std=c++11 $CFLAGS" and resume the installation.

```bash
# Clone the repository
git clone https://github.com/bsc-dd/hecuba.git hecuba_repo
cd hecuba_repo

# To install hecuba to the default directory
python setup.py install
```

To install Hecuba to the user space, or a custom directory run:

```bash
# Install to userspace, under $HOME/.local
python setup.py install --user

# Install to a user-defined path $CUSTOM_PATH
python setup.py install --prefix=$CUSTOM_PATH
```

The following option allows to specify the path where compiled dependencies should be installed alongside their headers. Useful when compiling a code against the Hecuba C++ core.

```bash
# Install Hecuba dependencies to custom path: $HOME/.local
python3 setup.py install --prefix=$HECUBA_ROOT --c_binding=$HOME/.local
```

This will:
- Install the Hecuba dependencies (libuv, cassandra driver and TBB) under $HOME/.local
- Install the Python package under $HECUBA_ROOT



### Install without Internet:

By running the install process on a computer with internet access, the dependencies will be downloaded as tar balls under `hecuba_core/dependencies`. Then, it is only necessary to copy the dependencies to the remote computer under the same directory. 


### Install the Hecuba core only

In some circumstances, it is useful to use the Hecuba core to interface Cassandra with C++ applications. In this case, the installation is performed completely manual (still). The following commands build the Hecuba core:

```bash
cmake -Hhecuba_core -Bbuild
make -C build
```
And finally, under the "build" folder, we will find the subfolders "include" and "lib" which need to be moved to the desired installation path.


# 3rd party software:

Cassandra database. [Github](https://github.com/apache/cassandra). Version 3.10 or later. Desirable 3.11.4 or later.


## Auto-downloaded

They are automatically downloaded if they can not be located in the system by CMake.

* LIBUV, requisite for Cassandra C++ driver. [Github](https://github.com/libuv/libuv). Version 1.11.0 or later.

* Datastax C++ Driver for apache cassandra. [Github](https://github.com/datastax/cpp-driver), [Official](https://datastax.github.io/cpp-driver/). Version 2.5.0 or later.

* TBB, Intel Threading Building Blocks, concurrency & efficiency support. [Github](https://github.com/01org/tbb), [Official](https://www.threadingbuildingblocks.org). Version 4.4.


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
