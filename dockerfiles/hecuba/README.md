# Hecuba container

## Build
```bash
cd ${HECUBA_ROOT}
docker build -t hecuba:0.1 dockerfiles/hecuba
```

## Launch

Create a shared network between Cassandra nodes and application nodes
```bash
docker network create --driver bridge cassandra_bridge
```

Launch and get the docker id of one Cassandra node
```bash
CASSANDRA_ID=$(docker run --rm --network=cassandra_bridge -d  cassandra)
```

Ask for the assigned ip
```bash
CASSANDRA_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ${CASSANDRA_ID})
```

Launch Hecuba
```bash
docker run --rm --env CONTACT_NAMES=${CASSANDRA_IP} --network=cassandra_bridge -it  hecuba:0.1 bash
```

