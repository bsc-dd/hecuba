.. _conf_param:

Hecuba Configuration Parameters
===============================

There are several parameters that can be defined when running our application. The basic parameters are the following:



* CONTACT_NAMES (default value: 'localhost'): list of the Storage System nodes separated by a comma (example: export CONTACT_NAMES=node1,node2,node3)

* NODE_PORT (default value: 9042): Storage System listening port

* EXECUTION_NAME (default value: "my_app’): Default name for the upper level in the app namespace hierarchy

* CREATE_SCHEMA (default value: True): if set to True, Hecuba will create its metadata structures into the storage system. Notice that these metadata structures are kept from one execution to another so it is only necessary to create them if you have deployed from scratch the storage system.


Hecuba Advanced Configuration Parameters
****************************************

* NUMBER_OF_BLOCKS (default value: 1024): Number of partitions in which the data will be divided for each node

* CONCURRENT_CREATION (default value: False): you should set it to True if you need to support concurrent persistent object creation. Setting this variable slows-down the creation task so you should keep it to False if only sequential creation is used or if the concurrent creation involves disjoint objects

* LOAD_ON_DEMAND (default value: True): if set to True data is retrieved only when it is accessed. If it is set to False data is loaded when an instance to the object is created. It is necessary to set to True if you code uses those functions of the numpy library that do not use the interface to access the elements of the numpy ndarray.

* DEBUG (default value: False): if set to True Hecuba shows during the execution of the application some output messages describing the steps performed

* SPLITS_PER_NODE (default value: 32): Number of partitions that generates the split method

* MAX_CACHE_SIZE (default value: 1000): Size of the cache. You should set it to 0 (and thus deactivate the utilization of the cache) if the persistent objects are small enough to keep them in memory while they are in used

* PREFETCH_SIZE (default value: 10000): Number of elements read in advance when iterating on a persistent object

* WRITE_BUFFER_SIZE (default value: 1000): size of the internal buffer used to group insertions to reduce the number of interactions with the storage system

* WRITE_CALLBACKS_NUMBER (default value: 16): number of concurrent on-the-fly insertions that Hecuba can support

* REPLICATION_STRATEGY (default value: 'SimpleStrategy'): Strategy to follow in the Cassandra database

* REPLICA_FACTOR (default value: 1): The amount of replicas of each data available in the Cassandra cluster

Hecuba Specific Configuration Parameters for the *storage_props* file of PYCOMPSs
*********************************************************************************

* CONTACT_NAMES (default value: empty): If this variable is set in the storage_props file, then COMPSs assumes that the variable contains the list of of an already running Cassandra cluster. If this variable is not set in the storage_props file, then the enqueue_compss command will use the Hecuba scripts to deploy and launch a new Cassandra cluster using all the nodes assigned to workers.

* RECOVER (default value: empty): if this variable is set in the storage_props file, then the enqueue_compss command will use the Hecuba scripts to deploy and launch a new Cassandra cluster starting from the snapshot identified by the variable. Notice that in this case, the number of nodes used to generate the snapshot should match the number of workers requested by the enqueue_compss command.

* MAKE_SNAPSHOT (default value: 0): the user should set this variable to 1 in the storage_props file if a snapshot of the database should be generated and stored once the application ends the execution (this feature is still under development, users can currently generate snapshots of the database using the c4s tool provided as part of Hecuba).
