
IMAGE=bsc_dd/hecubacassandra

.PHONY: build run net bash cass
run:
	docker run --rm --name cassandra_hecuba -d ${IMAGE}
	#docker run --rm --name cassandra_hecuba --network=cassandra_net -d ${IMAGE}
	sleep 5
	docker exec -it cassandra_hecuba systemctl is-enabled cassandra.service
	docker exec -it cassandra_hecuba systemctl enable cassandra.service
	docker exec -it cassandra_hecuba service cassandra start
	@echo ==========================
	@echo ==== DOCKER IMAGE RUN ====
	@echo ==========================
	@echo Use \'make bash\' to connect to the container

cass:
	docker exec -it cassandra_hecuba systemctl is-enabled cassandra.service
	docker exec -it cassandra_hecuba systemctl enable cassandra.service
	docker exec -it cassandra_hecuba service cassandra start

net:
	docker network create --attachable --driver bridge cassandra_net

build:
	docker build -t ${IMAGE} .
	@echo ============================
	@echo ==== DOCKER IMAGE BUILT ====
	@echo ============================
	@echo Use \'make run\' to run container

bash:
	docker exec -it cassandra_hecuba /bin/bash
