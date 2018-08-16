# Hecuba
Hecuba connector with databases implementing CQL language.


## Installation procedure

### Software requisites:

+ GCC 4.8 & up. Tested with versions 4.8.2, 4.8.5, 4.9.1.
+ CMake 3.3 & up. Tested with versions 3.5.0, 3.6.x, 3.7.1.
+ Libtools. Tested with versions 2.4.2


### 3rd party software:
They are automatically downloaded if they can not be located in the system by cmake.

* Cassandra database. [Github](https://github.com/apache/cassandra). Version 3.10 or later.

* LIBUV, requisite for Cassandra C++ driver. [Github](https://github.com/libuv/libuv). Version 1.11.0

* Datastax C++ Driver for apache cassandra. [Github](https://github.com/datastax/cpp-driver), [Official](https://datastax.github.io/cpp-driver/). Version 2.5.0

* TBB, Intel Threading Building Blocks, concurrency & efficiency support. [Github](https://github.com/01org/tbb), [Official](https://www.threadingbuildingblocks.org). Version 4.4



### Instructions to install:

cmake -H. -Bbuild

make -C build

make -C install

The libraries will be located in ${current path}/_instal/lib.

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
