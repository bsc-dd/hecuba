# Hecuba ![](https://travis-ci.org/bsc-dd/hecuba.svg?branch=master) [![codecov](https://codecov.io/gh/bsc-dd/hecuba/branch/master/graph/badge.svg)](https://codecov.io/gh/bsc-dd/hecuba)
Non-relational databases are nowadays a common solution when dealing with huge data set and massive query work load. These systems have been redesigned from scratch in order to achieve scalability and availability at the cost of providing only a reduced set of low-level functionality, thus forcing the client application to take care of complex logics. As a solution, our research group developed **Hecuba**, a set of tools and interfaces, which aims to facilitate programmers with an efficient and easy interaction with non-relational technologies.

## Installation procedure

### Software requisites:

+ GCC 4.8 & up. Tested with versions 4.8.2, 4.8.5, 4.9.1, 5.4.0, 7.2.0, 7.2.1, 7.3.1.
+ Intel Compiler 17.0.4 & up.
+ CMake 3.3 & up. Tested with versions 3.5.0, 3.6.x, 3.7.1, 3.9.3.
+ Python > 3.4 with the development package installed (includes the Python dynamic library). Tested with versions ranging from 3.4 to 3.7.
+ Python modules: setuptools, numpy, cassandra-driver (ccmlib, unittest and mock for testing).


### OpenSuse
Requirements on OpenSuse 42.2
```
sudo zypper install cmake python-devel gcc-c++ python-numpy-devel
```


### 3rd party software (autodownloaded):
They are automatically downloaded if they can not be located in the system by cmake.

* Cassandra database. [Github](https://github.com/apache/cassandra). Version 3.10 or later. Desirable 3.11.4 or later.

* LIBUV, requisite for Cassandra C++ driver. [Github](https://github.com/libuv/libuv). Version 1.11.0 or later.

* Datastax C++ Driver for apache cassandra. [Github](https://github.com/datastax/cpp-driver), [Official](https://datastax.github.io/cpp-driver/). Version 2.5.0 or later.

* TBB, Intel Threading Building Blocks, concurrency & efficiency support. [Github](https://github.com/01org/tbb), [Official](https://www.threadingbuildingblocks.org). Version 4.4.


### Instructions to install:

A file named `setup.py` should be present inside the root folder. By running the command `python setup.py install` the application will be installed to the system. However, a more versatile install is produced by adding `--user` to the previous command which will install the application in the user space, thus not requiring privileges.

This procedure will launch a cmake process to build a submodule of the application producing a lot of output, which is completely normal. It may occur that CMake selects the wrong compiler, which might not support C++11. In this case, the environment flags `CC` and `CXX` should be defined to point to the C and C++ compilers respectively and the installing command relaunched. Some old compilers fail if the following environment variable `CFLAGS="-std=c++11 $CFLAGS" is not defined.

Bear in mind that for being able to use Numpy arrays, the Numpy developer package should be present on the system. It contains all the necessary headers.


#### Install without Internet:

Perform a local install, and then copy all the files excluding the folder "Build". Most of the dependencies will be already downloaded under hecuba_core/dependencies.


#### Install the Hecuba core only

In some circumstances, it is useful to use the Hecuba core to interface Cassandra with C++ applications. In this case, installation needs to be done manual (still). The following commands build the Hecuba core:

```bash
cmake -Hhecuba_core -Bbuild
make -C build
```
And finally, under the "build" folder we will find the subfolder "include" and "lib" which needs to be moved to the desired installation path.


### Instructions to execute with Hecuba:

+ Start Cassandra and configure the environment variables as described in [Wiki Launch Hecuba](https://github.com/bsc-dd/hecuba/wiki/1:-User-Manual#how-to-execute).

+ Append the path the site-packages directory where Hecuba was installed to the PYTHONPATH environment variable,


Please, refer to the [Hecuba manual](https://github.com/bsc-dd/hecuba/wiki/1:-User-Manual) to check more Hecuba configuration options.


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
