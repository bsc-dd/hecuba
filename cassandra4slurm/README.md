# cassandra4slurm

### Usage: 
. ./launcher.sh [ **-h** | **RUN** [ **-s** ] [ **N** ] | **RECOVER** [ **-s** ] | **STATUS** | **KILL** ]

IMPORTANT: The leading dot is needed for Hecuba (https://github.com/bsc-dd/hecuba) since this launcher sets some environment variables.

- `bash launcher.sh -h`	
	Prints this usage help.

- `bash launcher.sh RUN`
	Starts new a Cassandra Cluster. Starts `N` nodes, if given. Default is 4.
	Using the optional parameter `-s` it will save a snapshot after the execution.

- `bash launcher.sh RECOVER`
	Shows a list of snapshots from previous Cassandra Clusters and restores the chosen one.
	Using the optional parameter `-s` it will save a snapshot after the execution.

- `bash launcher.sh STATUS`
	Gets the status of the Cassandra Cluster.

- `bash launcher.sh KILL`
	If a Cassandra Cluster is running, it is killed, aborting the process.



### Installation:


1) Copy the files to  **$PATH/lib/cassandra4slurm**

2) Add symlinks:
```bash
${ROOT}/bin/c4s -> ${ROOT}/lib/cassandra4slurm/launcher.sh
${ROOT}/bin/cassandra4slurm -> ${ROOT}/lib/cassandra4slurm/launcher.sh
${ROOT}/bin/runapp -> ${ROOT}/lib/cassandra4slurm/execute.sh
```


3) Overall organization

-bin
-- cassandra4slurm launchers

-include
-- hecuba/*.[cpp/h]
-- tbb/..
-- uv/cassandra headers


-cassandra-d8tree (Cassandra including Qbeast)

-lib
-- libhfetch.so / hfetch.so / libcassandra.so / libuv.so / libtbb.so (Hecuba core libs)
-- cassandra4slurm/.. (cassandra4slurm scripts)
-- python2.7/site-packages/
---- cassandra-driver
---- futures-2.1.6
---- Hecuba-0.1

-src (Codes used for installation)

-tables_templates (Template schemas)
