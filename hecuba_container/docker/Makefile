
IMAGE=bsc_dd/hecubacassandra
# DOCKER: Docker engine to use (docker|podman)
DOCKER=docker

.PHONY: build run net bash cass
run:
	${DOCKER} run --rm --name cassandra_hecuba -d ${IMAGE}
	#${DOCKER} run --rm --name cassandra_hecuba --network=cassandra_net -d ${IMAGE}
	sleep 5
	${DOCKER} exec -it cassandra_hecuba systemctl is-enabled cassandra.service
	${DOCKER} exec -it cassandra_hecuba systemctl enable cassandra.service
	${DOCKER} exec -it cassandra_hecuba service cassandra start
	@echo ==========================
	@echo ==== DOCKER IMAGE RUN ====
	@echo ==========================
	@echo Use \'make bash\' to connect to the container

cass:
	${DOCKER} exec -it cassandra_hecuba systemctl is-enabled cassandra.service
	${DOCKER} exec -it cassandra_hecuba systemctl enable cassandra.service
	${DOCKER} exec -it cassandra_hecuba service cassandra start

net:
	${DOCKER} network create --attachable --driver bridge cassandra_net

build:
	${DOCKER} build -t ${IMAGE} .
	@echo ============================
	@echo ==== DOCKER IMAGE BUILT ====
	@echo ============================
	@echo Use \'make run\' to run container

bash:
	${DOCKER} exec -it cassandra_hecuba /bin/bash
