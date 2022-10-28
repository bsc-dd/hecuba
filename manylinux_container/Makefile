
DOCKER_IMAGE=bsc_dd/hecubacassandra
PLAT=manylinux2014_x86_64

.PHONY: build run bash

build:	## Build the docker image
	docker build -t ${DOCKER_IMAGE} .
	@echo ============================
	@echo ==== DOCKER IMAGE BUILT ====
	@echo ============================
	@echo Use \'make wheel\' to generate the wheels

wheel:	## Generate the 'wheel' files
	docker container run --rm -it -e PLAT=${PLAT}     -v "${PWD}/..":/io     -e CMAKE_PARALLEL_BUILD=4 "${DOCKER_IMAGE}" /io/manylinux_container/build-wheels.sh

bash:	## Open a bash session to the container (Testing)
	docker container run --rm -it -e PLAT=${PLAT}     -v "${PWD}/..":/io     -e CMAKE_PARALLEL_BUILD=4 "${DOCKER_IMAGE}" /bin/bash

test: ## Upload wheels to 'testPypi'
	python3 -m pip install --upgrade twine
	python3 -m twine upload --repository testpypi ../wheelhouse/Hecuba*

upload: ## Upload wheels to 'Pypi'
	python3 -m pip install --upgrade twine
	python3 -m twine upload ../wheelhouse/Hecuba*
