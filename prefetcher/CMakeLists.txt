cmake_minimum_required(VERSION 3.3)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -fPIC")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")


list(APPEND CMAKE_MODULE_PATH cmake)
set(CMAKE_POSITION_INDEPENDENT_CODE YES)

#include("cmake/HunterGate.cmake")
#HunterGate(
#        URL "https://github.com/ruslo/hunter/archive/v0.17.19.tar.gz"
#        SHA1 "6be30333a0d94d84687a318c127391991d551de8"
#)



project(prefetcher)

#hunter_add_package(PocoCpp)
#find_package(Poco REQUIRED Foundation CONFIG)

#set_property(TARGET Poco::Foundation PROPERTY POSITION_INDEPENDENT_CODE ON)

#hunter_add_package(GTest)
#find_package(GTest CONFIG REQUIRED)
#hunter_add_package(Cassandra)

include(ExternalProject) # ExternalProject_Add
message(installing libs into ${CMAKE_CURRENT_LIST_DIR}/_install)

ExternalProject_Add(
    PocoALL
    URL https://pocoproject.org/releases/poco-1.7.7/poco-1.7.7.tar.gz
    URL_HASH SHA1=bdac2ff5cb2bea078cc71f1d85704bbeba7e14c9
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_LIST_DIR}/_install -DENABLE_CPPUNIT=OFF -DENABLE_XML=OFF -DENABLE_JSON=OFF -DENABLE_MONGODB=OFF -DENABLE_PDF=OFF -DENABLE_UTIL=OFF -DENABLE_NET=OFF -DENABLE_NETSSL=OFF -DENABLE_NETSSL_WIN=OFF -DENABLE_CRYPTO=OFF -DENABLE_DATA=OFF -DENABLE_DATA_SQLITE=OFF -DENABLE_DATA_MYSQL=OFF -DENABLE_DATA_POSTGRESQL=OFF -DENABLE_DATA_ODBC=OFF -DENABLE_SEVENZIP=OFF -DENABLE_ZIP=OFF -DENABLE_APACHECONNECTOR=OFF -DENABLE_CPPPARSER=OFF -DENABLE_POCODOC=OFF -DENABLE_PAGECOMPILER=OFF -DENABLE_PAGECOMPILER_FILE2PAGE=OFF -DENABLE_REDIS=OFF -DENABLE_TESTS=OFF -DENABLE_SAMPLES=OFF -DENABLE_MSVC_MP=OFF
)

FIND_LIBRARY(Poco NAMES PocoFoundation PATHS ${CMAKE_CURRENT_LIST_DIR}/_install/lib)


ExternalProject_Add(
        Cassandra
        URL https://github.com/datastax/cpp-driver/archive/2.5.0.tar.gz
        URL_HASH SHA1=70036c9d9685daf37ccec95851e1b1823357b4ec
        CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_LIST_DIR}/_install -DCASS_USE_LIBSSH2=OFF -DCASS_USE_OPENSSL=OFF
)
FIND_LIBRARY(cass NAMES cassandra PATHS ${CMAKE_CURRENT_LIST_DIR}/_install/lib)
#set(Css ${CMAKE_CURRENT_LIST_DIR}/_install/lib/libcassandra_static.a)
#hunter_add_package(TBB)


find_package(TBB) # searches for the system TBB

if (NOT TBB_FOUND)
    message(Downloading our own TBB)
    ExternalProject_Add(
            tbb2017
            URL "https://www.threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb2017_20161128oss_src.tgz"
            CMAKE_COMMAND echo
            BUILD_COMMAND make tbb
            BUILD_IN_SOURCE 1
            INSTALL_COMMAND cp -r build/linux_intel64_gcc_cc4.8_libc2.19_kernel4.1.12_release/  ${CMAKE_CURRENT_LIST_DIR}/_install/lib
    )
#    find_package(TBB NAMES tbb PATHS ${CMAKE_CURRENT_LIST_DIR}/_install/lib)
endif()


find_package(TBB NAMES tbb2017 PATHS ${CMAKE_CURRENT_LIST_DIR}/_install/lib)




find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS} ${CMAKE_CURRENT_LIST_DIR}/_install/lib)

#End of dependencies



set(SOURCE_FILES Prefetch.cpp HCache.cpp CacheTable.cpp TupleRow.cpp TupleRowFactory.cpp TupleRowFactory.h)



add_library(prefetcher SHARED ${SOURCE_FILES})

target_include_directories(prefetcher PRIVATE ${TBB_INCLUDE_DIRS} ${CMAKE_CURRENT_LIST_DIR}/_install/include/ )
target_compile_definitions(prefetcher PRIVATE ${TBB_DEFINITIONS})


target_link_libraries( prefetcher PocoFoundation ${css} ${PYTHON_LIBRARIES} ${TBB})


#FIND_LIBRARY(PT NAMES pthread  PATHS /lib)





