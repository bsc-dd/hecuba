.. _exe_pycompss:

Containerized execution with PyCOMPSs
=====================================

Hecuba offers support to execute Cassandra in a singularity container.
At this moment, this support is only available through the execution with PyCOMPSs.
To activate this type of execution, the user has to set the enqueue_compss option *storage_container_image* to the value *default*.
This option requires a valid singularity Cassandra file to be in the directory $HECUBA_ROOT/singularity.
In the next release, we plan to extend this functionality to use this *storage_container_image* option to specify a different path for the singularity file.

.. code-block:: bash

    PATH_TO_COMPSS_INSTALLATION/enqueue_compss \
    --num_nodes = 4 \
    --storage_props = storage_props.cfg \
    --storage_home=$HECUBA_ROOT/compss/ \
    --scheduler=es.bsc.compss.scheduler.fifodatanew.FIFODataScheduler \
    --storage_container_image=default \
    --lang=python \
    $(pwd)/myapp.py
