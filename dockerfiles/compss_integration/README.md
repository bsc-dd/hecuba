# Hecuba - COMPSs integration container
By default launches the integration tests

## Build
```bash
cd ${HECUBA_ROOT}
docker build -t hecompss:0.1 dockerfiles/compss_integration/
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

Launch Hecuba-COMPSs integration container
```bash
docker run --rm --env CONTACT_NAMES=${CASSANDRA_IP} --network=cassandra_bridge -v `pwd`:/io hecompss:0.1
```

