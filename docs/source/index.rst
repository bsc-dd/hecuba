.. Hecuba-read_the_docs documentation master file, created by
   sphinx-quickstart on Thu Jan 18 10:23:54 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Hecuba's user manual!
================================================

In this manual, we describe how to implement a Python application using `Hecuba <https://github.com/bsc-dd/hecuba/wiki>`_ and which are the main features that Hecuba implements to boost the performance of an application using Cassandra. Hecuba developers can improve their productivity as Hecuba implements all the necessary code to access the data. Thus, applications can access data as if it was in memory and Hecuba translates this code at runtime to access the underlying storage system. Also, Hecuba implements some optimizations to favor data locality and to reduce the number of interactions with the backing storage and thus, to speedup the accesses to data.

.. image:: images/HecubaBSCLogo2_1.png

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   applications
   deployment
   basics
   pycompss
   execution
   exe_pycompss
   conf_param
   cplusplus
   lambda

..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
