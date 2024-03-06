.. _applications:

Hecuba applications at a glance
===============================

One of the goals of Hecuba is to provide programmers with an easy and portable interface to access data. This interface is independent of the type of system and storage used to keep data, enhancing the portability of the applications. To sum up, using Hecuba the applications can access data like regular objects stored in memory and Hecuba translates the code at runtime into the proper code, according to the backing storage used in each scenario.

The current implementation of Hecuba handles in-memory objects, or persistent data storage provided by Apache Casssandra databases. This chapter guides how to create a Python application that uses Hecuba object abstractions to handle data persistence.

We will start by defining a set of classes that represent the persistent data. The user must inherit from one of the main abstractions provided by Hecuba, thus, a *StorageObj* or a *StorageDict*. Programmers can also use *StorageNumpy* Hecuba class to instantiate persistent numpy ndarrays.

The StorabeObj allows the user to define persistent attributes, accessed with the python object protocol. On the other hand, the StorageDict behaves like a python dictionary, accepting a key to identify the values. In both cases, the in-memory and persistent data will be handled transparently.

Next, the user must define the data model with the concrete data types that will be stored in a persistent layer. The specification is written as a Python comment, and the structure differs if we inherit from a *StorageObj* or a *StorageDict*. For instance, to define a set of attributes we will use the ``@ClassField`` with a *StorageObj*, and to define a dictionary the ``@TypeSpec`` with a *StorageDict*.

.. code-block:: python

   from hecuba import StorageObj
   import numpy as np

   class Dataset(StorageObj):
       '''
       @ClassField author str
       @ClassField open_access bool
       @ClassField injected_particles int
       @ClassField geometry numpy.ndarray
       '''

The user data model expresses that there will be four attributes which must be stored onto the persistent layer, for each instance of the class. Also, by adding different ``@ClassField`` the user can define any number of persistent attributes. Once the class is defined, the application can instantiate as many objects as needed. On the other hand, we can have a *StorageDict* to store the some persistent results with the following definition:

.. code-block:: python

    from hecuba import StorageDict

    class Results(StorageDict):
        '''
        @TypeSpec dict <<particle_id:int, step:float>, x:double, y:double, z:double>
        '''

    experiment = Results("breathing.results")

And by mixing both definitions we can write a small application. Note that the ``Dataset`` class has an attribute ``particles`` referencing the class ``ParticlesPositions``.

.. code-block:: python

    from hecuba import StorageObj, StorageDict
    import numpy as np



    class ParticlesPositions(StorageDict):
        '''
        @TypeSpec dict <<particle_id:int>, x:double, y:double, z:double>
        '''


    class Dataset(StorageObj):
        '''
        @ClassField author str
        @ClassField open_access bool
        @ClassField injected_particles int
        @ClassField geometry numpy.ndarray
        @ClassField particles ParticlesPositions
        '''


    dt1 = Dataset("breathing.case235")

    dt1.author = "BSC"
    dt1.open_access = True
    dt1.injected_particles = 250000
    dt1.geometry = np.load("./geom_file.npy")

    for part_id in range(dt1.injected_particles):
        dt1.particles[part_id] = list(np.random.random_sample(3,))

By passing a name of type ``str`` to the initializer of a Hecuba class instance, the object becomes persistent and sends the data to the persistent storage. Said name will act as an identifier for its data and other objects created with the same name will access the same data. In this way, if we pass a name which was previously used to create an object we will retrieve the previously persisted data.

Initializing an instance of an hecuba class without a name results in a regular in-memory object. However, its data can be persisted at any moment by calling the instance method *make_persistent*, provided and implemented by all Hecuba classes.
This method expects a ``str`` name, in the same way the initializer did, and will be used to identify the data in the future. This method will send the data to the data store, mark the object as persistent, and, future accesses will access the data store if deemed necessary.

.. code-block:: python

    class ParticlesPositions(StorageDict):
        '''
        @TypeSpec dict <<particle_id:int>, x:double, y:double, z:double>
        '''

    r=ParticlesPositions()

    r.make_persistent("OutputData")
