Cassandra SINGULARITY image construction in MN4
===============================================

If you have access to internet in MN4 then you can create the cassandra singularity sandbox as:
'''
singularity build --sandbox cassandra docker://cassandra
'''

Otherwise, go to a machine with access to internet, download the cassandra docker image and
save the Cassandra docker image to a tar file:
'''
docker save 5b647422e184 -o cassandra.tar
'''

Move the generated cassandra.tar to MN4:
'''
scp cassandra.tar bsc31226@mn1.bsc.es:.
'''

Build a singularity sandbox from the Cassandra.tar file:
'''
singularity build --sandbox cassandra docker-archive://cassandra.tar
'''

This 'cassandra' sandbox must be installed in *$HECUBA_ROOT/singularity*

This allows to use enqueue\_compss to launch COMPSs applications using Hecuba in containers (we assume that both COMPSs and Hecuba are available in the machine).
