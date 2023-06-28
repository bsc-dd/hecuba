
# DOCKER: Docker engine to use (docker|podman)
DOCKER=docker
DOCKER_IMAGE=bsc_dd/hecubacassandra
PLAT=manylinux2014_x86_64

.PHONY: build run bash

build:	## Build the docker image
	${DOCKER} build -t ${DOCKER_IMAGE} .
	@echo ============================
	@echo ==== DOCKER IMAGE BUILT ====
	@echo ============================
	@echo Use \'make wheel\' to generate the wheels

../storageAPI/storageItf/target/StorageItf-1.0.jar: ../storageAPI/storageItf/src/main/java/storage/StorageItf.java
	@echo ============================
	@echo [WARNING] Unable to find [ $@ ]
	@echo 	Trying to regenerate it
	@echo ============================
	cd ../storageAPI/storageItf && mvn org.apache.maven.plugins:maven-assembly-plugin:assembly

wheel:	 ../storageAPI/storageItf/target/StorageItf-1.0.jar ## Generate the 'wheel' files
	${DOCKER} container run --rm -it -e PLAT=${PLAT}     -v "${PWD}/..":/io     -e CMAKE_PARALLEL_BUILD=32 "${DOCKER_IMAGE}" /io/manylinux_container/build-wheels.sh

bash:	## Open a bash session to the container (Testing)
	${DOCKER} container run --rm -it -e PLAT=${PLAT}     -v "${PWD}/..":/io     -e CMAKE_PARALLEL_BUILD=32 "${DOCKER_IMAGE}" /bin/bash

test: ## Upload wheels to 'testPypi'
	python3 -m pip install --upgrade twine
	python3 -m twine upload --repository testpypi ../wheelhouse/Hecuba*

upload: ## Upload wheels to 'Pypi'
	python3 -m pip install --upgrade twine
	python3 -m twine upload ../wheelhouse/Hecuba*
