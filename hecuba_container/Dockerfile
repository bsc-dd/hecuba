FROM ubuntu:18.04

RUN apt-get update &&  apt-get install -y \
        software-properties-common \
  && add-apt-repository ppa:deadsnakes/ppa

# Hecuba requirements
RUN apt-get update \
    && apt-get install -y \
        apt-transport-https \
        ca-certificates \
        gnupg \
        software-properties-common \
        wget \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - \
    && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' \
    && apt-get update \
# Hecuba dependencies
    && apt-get install -y --no-install-recommends \
# Install Packages
        cmake \
        maven \
        git \
# Runtime dependencies
        openjdk-8-jdk \
# Bindings-common-dependencies
        libtool automake build-essential \
# C-binding dependencies
        libboost-all-dev \
# Python-binding dependencies
        python3-dev python3-pip python3-setuptools \
    && pip3 install wheel dill decorator coverage numpy==1.16 ipython==7.9.0

RUN apt-get install libuv1  # Required as UBUNTU 18.04 and libuv are not good friends, otherwise setting LD_LIBRARY_PATH is required

# Define Hecuba Environment variables
ENV CMAKE_BUILD_PARALLEL_LEVEL=4

# Hecuba version
RUN cd / \
    && git clone https://github.com/bsc-dd/hecuba.git hecuba \
    && cd hecuba \
    && python3 setup.py install \
    && cd /hecuba/storageAPI/storageItf \
    && mvn assembly:assembly

# Cassandra
RUN apt-get install -y curl \
    && echo "deb http://www.apache.org/dist/cassandra/debian 40x main" | tee -a /etc/apt/sources.list.d/cassandra.sources.list \
    && curl -s https://downloads.apache.org/cassandra/KEYS | apt-key add - \
    && apt-get update  \
    && DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata \
    && apt-get install -y cassandra

# SSH
RUN apt-get install -y ssh \
    && service ssh start

# Expose SSH port and run SSHD (to allow container to stay alife)
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]
