.. _execution:

How to execute
==============

Sequential applications
***********************

To run a sequential application using Hecuba is really simple after the requirements have been satisfied. We just need to execute the application with python, and it will use the configured data store:

.. code-block:: bash

    python3 myApp.py

This command assumes that you have a running a local Cassandra instance.
If you need to connect to a different instance of Cassandra you must set the variables *CONTACT_NAMES* and *NODE_PORT* with the suitable values
(see `Hecuba Configuration Parameters <https://github.com/bsc-dd/hecuba/wiki/1%3A-User-Manual#hecuba-configuration-parameters>`_).

Parallel applications on queue based systems
********************************************

To run a parallel Hecuba application using PyCOMPSs you should execute the *enqueue_compss* command setting the options storage_props and *storage_home*.
The *storage_props* option is mandatory and should contain the path of an existing file.
This file can contain all the Hecuba configuration options that the user needs to set (can be an empty file).
The *storage_home* option contains the path to the Hecuba implementation of the Storage API required by COMPSs.
Following, we show an example of how to use PyCOMPSs and Hecuba to run the python application in the file myapp.py.
In this example, we ask PyCOMPSs to allocate 4 nodes and to use the scheduler that enhances data locality for tasks using persistent objects.
We assume that the variable *HECUBA_ROOT* contains the path to the installation directory of Hecuba.

.. code-block:: bash

    PATH_TO_COMPSS_INSTALLATION/enqueue_compss \
    --num_nodes = 4 \
    --storage_props = storage_props.cfg \
    --storage_home=$HECUBA_ROOT/compss/ \
    --scheduler=es.bsc.compss.scheduler.fifodatanew.FIFODataScheduler \
    --lang=python \
    $(pwd)/myapp.py