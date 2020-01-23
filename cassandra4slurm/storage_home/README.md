# Storage API scripts for enqueue_compss

## Known issues
- the `storage_props` file should be created for each execution on the user's folder or tmp


## Files

`storage_init.sh` and `storage_stop.sh` are called by COMPSs when using the `enqueue_compss` command.

`storage_opts.txt` contains the storage options.


They must be placed under a folder called **scripts**.


## Command

To execute with `enqueue_compss` one can do:

```bash
export PYTHONPATH=/path/to/application:$PYTHONPATH
export HECUBA_STORAGE=/path/to/hecuba_storage
# Where hecuba_storage contains a folder named "scripts" with the storage_init.sh and storage_stop.sh

export CLASSPATH=/path/to/StorageItf.jar:$CLASSPATH

enqueue_compss --qos=debug -d --pythonpath=$PYTHONPATH --python_interpreter=python3 --storage_home=$HECUBA_STORAGE --storage_props=$HECUBA_STORAGE/scripts/storage_opts.txt --classpath=$CLASSPATH
```

