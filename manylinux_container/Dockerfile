#FROM quay.io/pypa/manylinux1_x86_64 DEPRECATED!
FROM quay.io/pypa/manylinux2014_x86_64

# Define Hecuba Environment variables
ENV CMAKE_BUILD_PARALLEL_LEVEL=4

# Install a system package required by our library
RUN yum -y upgrade

RUN yum install -y  java #openssl openssl-devel

# Yum installs an old version of CMAKE, therefore use a more modern one... take
# into account that glibc is 2.5 (aka 'modern' cmake uses 2.6... T_T therefore,
# searching for a working one... cmake-3.4.0 fits the bill)
RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.4.0/cmake-3.4.0-Linux-x86_64.tar.gz | tar -xz \
    && cd cmake-3.4.0-Linux-x86_64 \
    && cp -r * /usr/ \
    && cd .. \
    && rm -rf cmake-3.4.0-Linux-x86_64

# getting the latest libuv
RUN curl -L https://github.com/libuv/libuv/archive/v1.19.2.tar.gz|tar -xz \
    && cd libuv-1.19.2/\
    && sh autogen.sh\
    && ./configure\
    && make install \
    && cd .. \
    && rm -rf libuv-1.19.2/


# getting the latest Cassandra cpp driver
RUN cd; curl -L https://github.com/datastax/cpp-driver/archive/2.14.1.tar.gz|tar -xz \
    && cd cpp-driver-2.14.1/ \
    && cmake -H. -Bbuild -DCASS_USE_LIBSSH2=OFF -DCASS_USE_OPENSSL=OFF \
    && cd build;make; make install

# Copy needed header files
RUN cp /root/cpp-driver-2.14.1/src/murmur3.hpp /usr/local/include \
    && cp /root/cpp-driver-2.14.1/src/macros.hpp /usr/local/include \
    && cd; rm -rf cpp-driver-2.14.1


# yaml-cpp
RUN cd; curl -L https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.7.0.tar.gz|tar -xz\
    && cd yaml-cpp-yaml-cpp-0.7.0\
    && cmake -H. -Bbuild -DYAML_BUILD_SHARED_LIBS=ON\
    && cd build; make; make install \
    && cd ; rm -rf yaml-cpp-yaml-cpp-0.7.0

# kafka
RUN cd; curl -L https://github.com/edenhill/librdkafka/archive/refs/tags/v1.9.2.zip --output librdkafka-1.9.2.zip \
    && unzip librdkafka-1.9.2.zip \
    && cd librdkafka-1.9.2/ \
    && ./configure \
    && make; make install \
    && cd; rm -rf  librdkafka-1.9.2/ ; rm -rf librdkafka-1.9.2.zip

# BOOST
RUN cd; curl -L https://boostorg.jfrog.io/artifactory/main/release/1.67.0/source/boost_1_67_0.tar.gz|tar -xz \
    && cd boost_1_67_0/ \
    && ./bootstrap.sh

RUN cd /root/boost_1_67_0/ \
    && ./b2 --without-python install \
    && cd; rm -rf boost_1_67_0/



# ARROW
RUN cd; curl -L https://archive.apache.org/dist/arrow/arrow-0.15.1/apache-arrow-0.15.1.tar.gz|tar -xz\
    && cd apache-arrow-0.15.1/ \
    && cmake cpp -DCMAKE_INSTALL_LIBDIR=lib -DARROW_BUILD_STATIC=false \
    && sed -i s/_static//g src/arrow/CMakeFiles/arrow_shared.dir/link.txt \
    && make \
    && make  install \
    && cd; rm -rf apache-arrow-0.15.1/

# Hecuba version
#RUN cd / \
#    && git clone https://github.com/bsc-dd/hecuba.git hecuba \
#    && cd hecuba
#    && cd /hecuba/storageAPI/storageItf \
#    && mvn assembly:assembly

