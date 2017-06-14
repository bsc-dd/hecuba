# Hecuba

## Installation procedure

### Software requisites:

+ GCC 4.8 & up. GCC 6 not supported. Tested with versions 4.8.2, 4.8.5, 4.9.1.
+ CMake 3.3 & up. Tested with versions 3.5.0, 3.6.x, 3.7.1.
+ Libtools. Tested with versions 2.4.2
+ Python 2, starting from 2.7.5. Tested with versions 2.7.5 to 2.7.13. Python 3 not supported.
+ Python module: distutils

### Setting up the environment:

**Marenostrum III:**

```
module unload intel/13.0.1
module unload openmpi/1.8.1
module unload transfer/1.0
module unload bsc/current
module load gcc/4.9.1
module load PYTHON/2.7.12
module load CMAKE/3.5.0

export CC=gcc
export CXX=g++-4.9.1

python setup.py install --user
```
(Will produce warnings about conflicts with includes from gcc 4.3 (/usr/include) this path is hardcoded into CMake itself) 


**Juron:**

```
module load cmake/3.6.1
```


### 3rd party software:
They are automatically downloaded if they can not be located in the system by cmake.

* LIBUV, requisite for Cassandra C++ driver. [Github](https://github.com/libuv/libuv). Version 1.11.0

* Datastax C++ Driver for apache cassandra. [Github](https://github.com/datastax/cpp-driver), [Official](https://datastax.github.io/cpp-driver/). Version 2.5.0

* POCO, C++ libraries which implement a cache. [Github](https://github.com/pocoproject/poco/), [Official](https://pocoproject.org). Version 1.7.7

* TBB, Intel Threading Building Blocks, concurrency & efficiency support. [Github](https://github.com/01org/tbb), [Official](https://www.threadingbuildingblocks.org). Version 4.4



### Instructions to install:

A file named `setup.py` should be present inside the root folder. By running the command `python setup.py install` the application will be installed to the system. However, a more versatile install is produced by adding `--user` to the previous command which will install the application in the user space, thus not requiring privileges.

This procedure will launch a cmake process which builds a submodule of the application producing a lot of output, which is completly normal. It may occur that the compiler picked by Cmake doesn't support C++11 which will stop the building procedure. In this case, the environment flags `CC` and `CXX` should be defined to point to the C and C++ compilers respectively and the installing command relaunched. At this point the installation should proceed and finish without producing more errors.

Bear in mind that for being able to use Numpy arrays, the Numpy developer package should be present on the system. It contains all the necessary headers.
