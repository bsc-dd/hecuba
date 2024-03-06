.. _cplusplus:

Access to Hecuba from C++
=========================

We have added a layer that allows the interaction with the Hecuba core layer, without using the Python interface. This C++ interface is work in progress and currently does not cover all the functionality implemented by Hecuba. In addition, we expect to improve the usability of the interface in the next releases.

Class declaration
*****************

Currenly the C++ interface of Hecuba supports three types of Hecuba Objects:

* StorageDict
* StorageObject
* StorageNumpy

Programmers can define any class inheriting from one of these classes. The method to define each class depends of the base class.

Classes derived from StorageDicts
---------------------------------

StorageDict is a templatized class with a template composed of three elements: the class of the key, the class of the value and the name of the derived class.

The class of the key is the templatized class KeyClass defined by Hecuba, where the template is composed of a variable number of types: one for each element of the key. The following sentences define an alias for three different key classes with different number of attributes of different types:

.. code-block:: cpp

    using Key1 as KeyClass<int32_t>
    using Key2 as KeyClass<int32_t, float>
    using Key3 as KeyClass<int32_t, float, std::string>

The first element of the key acts as partition key and the rest acts as clustering keys.

The class of the value is the class ValueClass also defined by Hecuba and also with a variadic template: one type for each element of the value. For example:

.. code-block:: cpp

    using Value1 as ValueClass <std::string>
    using Value2 as ValueClass <int32_t, int32_t, std::string>

The types of both keys and values can be basic types or other Hecuba derived classes.

Thus, the following sentence defines a StorageDict that has one integer as key and a string as value:

.. code-block:: cpp

    class myDictClass:public StorageDict<Key1,Value1,myDictClass> {

    // can be empty

    }

Classes derived from StorageObjects
-----------------------------------

Hecuba provides the macro HECUBA_ATTRS to define the persistent attributes of classes derived from StorageObjects. The parameters of this macro are a comma separated pairs composed by type of the attribute and the name of the attribute. The number of persistent attributes is currently limited to 1024. The following sentence defines a class derived from a Storage Object with three attributes: "attr1" of type uint64_t, "attr2" of type std::string and "attr3" of type int32_t:

.. code-block:: cpp

    class myObjClass:public StorageObject{

    HECUBA_ATTRS (
         uint64_t, "attr1",
         std::string, "attr2",
         int32_t, "attr3"
        )
    }

Classes derived from StorageNumpy
---------------------------------

At this moment, Hecuba does not support any specialization of the StorageNumpy, for this reason, the C++ interface of Hecuba does not support the definition of derived classes from StorageNumpy. Programmer can instantiate StorageNumpys without any additional class definition.

Supported basic types
---------------------

Currently the basic types that C++ Hecuba supports are:

* int32_t, float, double, std:string, bool, char, int64_t.

New objects instantiation
*************************

The object instantiation is as any regular object:

.. code-block:: cpp

    myObjClass o;
    myDictClass d;
    StorageNumpy sn;

The only consideration is that if the programmer implements in the derived class a constructor with parameters, that constructor should explicitly call the default constructor of the base class.

In the case of the instantiation of a StorageNumpy, Hecuba implements an additional constructor that allows to set the numpy data during the instantiation. The signature of this constructor is:

.. code-block:: cpp

    StorageNumpy::StorageNumpy(void *data, vector<uint32_t> metas);

And this constructor is invoked in a declaration like the following one:

.. code-block:: cpp

    StorageNumpy sn (data, metas);

Where data is a pointer to a memory region with the content of the numpy in C order. Keep in mind that current implementation of StorageNumpy only support numpy.ndarrays of elements of type float. And metas is a vector that contains the size of each dimension of the numpy.ndarray.

Persisting new objects
**********************

Once an object is instantiated, the first step is to persist it. The connection with the database is performed when the first Hecuba object is persisted. And this connection will be active during all the execution of the process. The method to persist an Hecuba object is make_persistent:

.. code-block:: cpp

    void make_persistent(std::string name)

The parameter the name of the persistent object (a string that will identify the persistent object in the database):

This operation will prepare in the database the tables that will contain the object data. The operation to persist a Hecuba object is the same for all types of Hecuba base object.

.. code-block:: cpp

    d.make_persistent("mydictname");

All the insertions in a persisted object are sent asynchronously to the database.

At this moment it is not possible to insert data in a volatile Hecuba object but in the next releases we will extend this functionality.

The method make_persistent also generates a python file with the class definition of the object (for StorageDict and StorageObjects), that can be used from any python code that needs to access this persistent object.

Retrieving already existing persistent objects
**********************************************

Hecuba implements the method getByAlias to connect with a previously persisted Hecuba Object:

.. code-block:: cpp

    void getByAlias(std::string name)

The parameter of this method is the name of the object (the one used in the persisting operation).

.. code-block:: cpp

    myDict d;

    d.getByAlias("mydictname");

Object Access
*************

The interface to access Hecuba objects depends on the Hecuba base class.

Accessing StorageDicts
----------------------

In the case of StorageDict the access is implemented with the same operator of C++ maps or vectors: Hecuba overrides the indexing operator ([])).

In the insertion operation, the user has to specify the element of type KeyClass that acts as the index, and the element of type ValueClass that needs to insert. For example, if Key is an alias for a KeyClass composed of two elements of type int32_t and Value an alias for a ValueClass composed of three elements of type int32_t, std::string, and float, the following sentence represents a valid insertion in d:

.. code-block:: cpp

    d[Key(1,2)] = Value(3,"hi",(float)3.14)

In the read operation, the indexing operator returns a Value object. To facilitate the extraction of each element of the Value we have implemented the same interface that offers the standard tuples of C++. The following sentence will return the first element of Value

.. code-block:: cpp

    Value v = d[Key(1,2)];
    int32_t v1 = Value::get<0>(v);
    std::string v2 = Value::get<1>(v);
    float v3 = Value::get<2>(v);

The C++ interface of Hecuba also implements an iterator on the keys of a persistent StorageDict. The following loop accesses all the elements of a StorageDict:

.. code-block:: cpp

    Key k;
    Value v;
    for(auto it = d.begin(); it != d.end(); it++) {
        k=*it;
        v=d[k];
    }

Accessing StorageObjects
------------------------

In the case of StorageObjects the operator to access the attributes is the same than to access attributes of regular C++ objects: Hecuba overrides the accessing operator (.). If the user instantiates a StorageObject with one attribute named attr1 of type int32_t, then the following sentence will assing 1 to the attribute attr1:

.. code-block:: cpp

    o.attr1 = 1;

And the following sentence will read attr1 from o:

.. code-block:: cpp

    int32_t v_read = o.attr1;

Accessing StorageNumpys
-----------------------

In the case of StorageNumpy the current implementation of C++ interface of Hecuba only supports the insertion of the whole numpy data. The insertion can be performed in two ways:

* During the instantiation of the StorageNumpy, using the constructor that receives both the data and the metadata.

* Using the method setNumpy. The signature of this method is:

.. code-block:: cpp

    StorageNumpy::setNumpy(void *data, vector<uint32_t>metadata)

Where data is a pointer to a memory region with the content of the numpy.ndarray in C order and meta is a vector with the size of each dimension of the numpy.ndarray.

Synchronization with disk
*************************

All the insertions in persistent Hecuba objects are sent asynchronously to the database. Hecuba guarantees that at the end of the session all the data will be up to date in the database. However, Hecuba offers a method to explicitly synchronize an object with the database. The signature of this method is the following:

.. code-block:: cpp

    void sync()

And can be used with any type o persistent Hecuba objects:

.. code-block:: cpp

    d.sync();

Nested Objects
**************

Both StorageDicts and StorageObjects can contain other Hecuba objects, using the class name to declare the attribute.

For example, a StorageDict indexed with integers and with values of type StorageNumpy could be defined as follows:

.. code-block:: cpp

    using Key as KeyClass<int32_t>;
    using Value as ValueClass<StorageNumpy>;
    class nestedDict <Key,Value,nestedDict>: public StorageDict{
    }

And then used as follows:

.. code-block:: cpp

    nestedDict nd;
    nd.make_persistent("nd_dict");
    Key k(0);
    StorageNumpy sn(data,metadata); //data and metadata are variables
    sn.make_persistent("mynumpy"); //initialized properly
    Value v(sn);
    nd[k]=v;

We can use this dictionary as an attribute of an object:

.. code-block:: cpp

    class nestedObject: public StorageObject{
    public:
        HECUBA_ATTRS(
            std::string, description,
            nestedDict, content
        )
    }

And use it:

.. code-block:: cpp

    nestedObject no;
    no.make_persistent("nested_object");
    no.description = "numpys generated today";
    no.content = nd;

Compiling C++ applications using Hecuba
***************************************

Assuming that the variable HECUBA_ROOT contains the path where Hecuba is installed, applications using the C++ interface of Hecuba should be compiled using the following compiling command:

.. code-block:: bash

    g++ -o  application \
        application.cpp \
        -std=c++11 \
        -I ${HECUBA_ROOT}/include \
        -I ${HECUBA_ROOT}/include/hecuba \
        -L${HECUBA_ROOT}/lib \
        -lhfetch \
        -Wl,-rpath,${HECUBA_ROOT}/lib \
        -Wl,-rpath,${HECUBA_ROOT}/../Hecuba.libs

This compilation command works both for the standalone C++ Hecuba installation and for the python Hecuba installation where the HECUBA_ROOT Variable should point to the directory that contains the python Hecuba package.