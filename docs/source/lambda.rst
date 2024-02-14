.. _lamba:

Lambda architecture in Hecuba
=============================

Hecuba facilitates the implementation of using both on-line and off-line processing by implementing a lambda architecture. With this architecture, the data is streamed concurrently with the data insertion in storage.

Currently we have implemented this feature for StorageDicts. The user has to define the class as a subclass of both StorageDict and StorageStream Hecuba classes. With this class definition, all the setitems will send asynchronously the data to the storage and synchronously through the stream.

Exemple: python specification (to be used from python code)

.. code-block:: python

    class mydict (StorageDict, StorageStream):
       '''
       @TypeSpec dict <<key:int>,val:numpy.ndarray>
       '''

Exemple: C++ class definition (to be used from c++ code)

.. code-block:: cpp

    class myDictClass <Key,Value,myDictClass>: public StorageDict, public
    StorageStream{

    }

The consumer should use the function poll to receive the streamed data. At this point of the implementation, poll is only supported by the Python interface of Hecuba.

.. code-block:: python

    def poll() # returns the key and the value of the streamed data

Example: consumer in Python

.. code-block:: python

    d=mydict("dictname")
    k,v=d.poll() #k will value 42 and v will be sn

The code for the producer is the same than when the streaming capability is not set.

Example: producer in Python

.. code-block:: python

    d=mydict("dictname")
    d[42]=sn # sn is a StorageNumpy

Example: producer in C++

.. code-block:: cpp

    myDictClass d;
    d.make_persistent("streamdict");

    // add here the declaration of k and v

    d[k] = v; //will store this item in the database
              // and will send it through the stream

Notice that the data is stored in the database as a regular StorageDict, so it is possible to implement applications to perform off-line analysis without activating the stream feature in the class definition.

A full example can be found in `Hecuba Streaming Examples <https://github.com/bsc-dd/hecuba/tree/master/examples/streaming>`_, containing a producer implemented both in Python and C++ and its corresponding consumer in Python.
