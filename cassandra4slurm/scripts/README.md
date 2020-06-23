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

Run:

```bash
$HECUBA_FOLDER/cassandra4slurm/install.sh $INSTALL_PATH
```

which will copy the scripts into $INSTALL_PATH/lib/cassandra4slurm and add symlinks to $INSTALL_PATH/bin

