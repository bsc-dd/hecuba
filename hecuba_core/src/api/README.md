C++ Interface user manual
----------------------------

Prerequisites
-------------
* Hecuba *must* have been installed specifying at least the '--c_binding'
  option with the install directory:

```
    $ python3 setup.py install --c_binding=/target/path/for/libs_and_headers
```

Assuming HECUBA_ROOT contains the path where Hecuba is installed
(/target/path/for/libs_and_headers),after the installation hecuba and its
dependency libraries can be found at $HECUBA_ROOT/lib and headers can be found
at $HECUBA_ROOT/include


Compiling your application with Hecuba
----------------------------------------
Your application must include the headers from hecuba:
```
    #include <hecuba/DataModel.h>
    #include <hecuba/HecubaSession.h>
    ...
    /* Use the C++ API */
```

To compile an application named apitest.cpp you should execute the following
command:

```
    $ g++ -o apitest \
        apitest.cpp  \
        -std=c++11 \
        -I ${HECUBA_ROOT}/include \
        -L${HECUBA_ROOT}/lib \
        -lhfetch \
        -Wl,-rpath,${HECUBA_ROOT}/lib
```


Executing your application
--------------------------
After defining any environment variable to configure Hecuba to contact with an
already running Cassandra instance, you may execute your application:

```
    $ ./apitest
```


