# Hecuba
Non-relational databases are nowadays a common solution when dealing with huge data set and massive query work load. These systems have been redesigned from scratch in order to achieve scalability and availability at the cost of providing only a reduce set of low-level functionality, thus forcing the client application to take care of complex logics. As a solution, our research group developed **Hecuba**, a set of tools and interfaces, which aims to facilitate programmers with an efficient and easy interaction with non-relational technologies.

## Installation procedure

### Software requisites:

+ GCC 4.8 & up. GCC 6 not supported. Tested with versions 4.8.2, 4.8.5, 4.9.1.
+ CMake 3.3 & up. Tested with versions 3.5.0, 3.6.x, 3.7.1.
+ Libtools. Tested with versions 2.4.2
+ Python 2, starting from 2.7.6. Python must be configured to support UCS-4. The dynamic library of python must be available. Tested with versions 2.7.5 to 2.7.13. Python 3 not supported.
+ Python modules: distutils, numpy, six, futures

### OpenSuse
Requirements on OpenSuse 42.2
```
sudo zypper install cmake python-devel gcc-c++ libtool python-numpy-devel
```


### 3rd party software:
They are automatically downloaded if they can not be located in the system by cmake.

* Cassandra database. [Github](https://github.com/apache/cassandra). Version 3.10 or later.

* LIBUV, requisite for Cassandra C++ driver. [Github](https://github.com/libuv/libuv). Version 1.11.0

* Datastax C++ Driver for apache cassandra. [Github](https://github.com/datastax/cpp-driver), [Official](https://datastax.github.io/cpp-driver/). Version 2.5.0

* POCO, C++ libraries which implement a cache. [Github](https://github.com/pocoproject/poco/), [Official](https://pocoproject.org). Version 1.7.7

* TBB, Intel Threading Building Blocks, concurrency & efficiency support. [Github](https://github.com/01org/tbb), [Official](https://www.threadingbuildingblocks.org). Version 4.4



### Instructions to install:

A file named `setup.py` should be present inside the root folder. By running the command `python setup.py install` the application will be installed to the system. However, a more versatile install is produced by adding `--user` to the previous command which will install the application in the user space, thus not requiring privileges.

This procedure will launch a cmake process which builds a submodule of the application producing a lot of output, which is completly normal. It may occur that the compiler picked by Cmake doesn't support C++11 which will stop the building procedure. In this case, the environment flags `CC` and `CXX` should be defined to point to the C and C++ compilers respectively and the installing command relaunched. At this point the installation should proceed and finish without producing more errors.

Bear in mind that for being able to use Numpy arrays, the Numpy developer package should be present on the system. It contains all the necessary headers.

### Instructions to execute with Hecuba:
The only requirement is to set the PYTHONPATH environment variable with the site-packages directory and the LD_LIBRARY_PATH enviroment variable with the path that contain the dynamic library of python.

Please, refer to the Hecuba manual to check more Hecuba configuration options.

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
