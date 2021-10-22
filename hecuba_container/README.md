Hecuba Container installation instructions
==========================================
The current Dockerfile builds a container with the Hecuba infrastructure and a
single Cassandra instance.

Requirements:
-------------
    - docker


Steps:
------
1) Build the container from the 'Dockerfile', setting a name (ex: IMAGE):
```
	docker build -t IMAGE .
```

2) When the previous command finishes, we have a usable container that we have to start:
```
	docker run --rm --name cassandra_hecuba -d IMAGE
```

3) When the container is started, we need to start cassandra service:
```
	docker exec -it cassandra_hecuba systemctl is-enabled cassandra.service
	docker exec -it cassandra_hecuba systemctl enable cassandra.service
	docker exec -it cassandra_hecuba service cassandra start
```
    [You may ignore the errors shown]

4) Now the container is ready, and you can connect to it starting a bash session:
```
	docker exec -it cassandra_hecuba /bin/bash
```

Alternativelly, you can use the makefile provided to automate these steps:
```
make build
make run
make bash
```
