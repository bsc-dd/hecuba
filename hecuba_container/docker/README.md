Hecuba Container installation instructions
==========================================
The current Dockerfile builds a container with the Hecuba infrastructure and a
single Cassandra instance.

Requirements:
-------------
    - docker
    - docker-compose

Description
-----------
The idea behind the container presented here is to simulate a Cassandra cluster. The way to do that is to create a scalable cassandra cluster using its own [Cassandra image](https://hub.docker.com/_/cassandra/). We use one container as the cassandra 'seed' (*cassandra_seed*) and we set  the variable CASSANDRA_SEEDS in any other node to add nodes to the cluster (*cassandra*).
Each cassandra container contains an hecuba installation. By default *cassandra_seed* shares the HOME directory inside the container using the same path (to ease the application execution shown later).


Start the cluster
-----------------
We use Docker-compose to launch the cassandra cluster (2 nodes by default):

~~~
    docker-compose up -d
~~~


or a single instance:

~~~
    docker-compose up -d --scale cassandra=0
~~~

or scale the cluster to 3 nodes (1 seed and 2 extra):

~~~
    docker-compose up -d --scale cassandra=2
~~~

But remember that these 'extra nodes' will run in the same phyisical machine, so plenty of memory is required.


Play with the cluster
---------------------
Now that the cluster is ready, you may connect to it and start playing:

~~~
	runhecubaapp-docker /bin/bash
~~~

The *runhecubaapp-docker* script gets the cassandra node's IPs (asking the *cassandra_seed*) and sets the required CONTACT_NAMES variable.


or you can run your own python application using its absolute PATH (located in some path under your HOME directory):

~~~
	runhecubaapp-docker python3 $PWD/your_own_app.py arg1 arg2 ...
~~~

(Here you can see the trick of sharing the HOME directory inside the same path in the directory)

Configuring environment
-----------------------
You may create a configuration file *hecuba_environment* in the *$HOME/.c4s/conf* path to export any environment variable you want to use inside the container and it will be sourced before your application.
