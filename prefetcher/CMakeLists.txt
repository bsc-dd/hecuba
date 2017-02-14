cmake_minimum_required(VERSION 3.3)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -fPIC")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_POSITION_INDEPENDENT_CODE YES)
project(hfetch)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include_directories( ${CMAKE_CURRENT_LIST_DIR}/_install/ /usr/include)
include(ExternalProject) # ExternalProject_Add

set(SOURCE_FILES Prefetch.cpp HCache.cpp CacheTable.cpp TupleRow.cpp TupleRowFactory.cpp)

add_library(hfetch SHARED ${SOURCE_FILES})



FIND_LIBRARY(POCO NAMES PocoFoundation PATHS ${CMAKE_CURRENT_LIST_DIR}/_install/lib)
FIND_LIBRARY(CASS NAMES cassandra PATHS ${CMAKE_CURRENT_LIST_DIR}/_install/lib)

message("DIR: " ${CMAKE_CURRENT_LIST_DIR}/_install/lib)


if(NOT POCO)
    message("Downloading our own POCO LIB " ${POCO})
    unset(POCO)
    ExternalProject_Add(
            POCO
            URL https://pocoproject.org/releases/poco-1.7.7/poco-1.7.7.tar.gz
            URL_HASH SHA1=bdac2ff5cb2bea078cc71f1d85704bbeba7e14c9
            CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_LIST_DIR}/_install -DENABLE_CPPUNIT=OFF -DENABLE_XML=OFF -DENABLE_JSON=OFF -DENABLE_MONGODB=OFF -DENABLE_PDF=OFF -DENABLE_UTIL=OFF -DENABLE_NET=OFF -DENABLE_NETSSL=OFF -DENABLE_NETSSL_WIN=OFF -DENABLE_CRYPTO=OFF -DENABLE_DATA=OFF -DENABLE_DATA_SQLITE=OFF -DENABLE_DATA_MYSQL=OFF -DENABLE_DATA_POSTGRESQL=OFF -DENABLE_DATA_ODBC=OFF -DENABLE_SEVENZIP=OFF -DENABLE_ZIP=OFF -DENABLE_APACHECONNECTOR=OFF -DENABLE_CPPPARSER=OFF -DENABLE_POCODOC=OFF -DENABLE_PAGECOMPILER=OFF -DENABLE_PAGECOMPILER_FILE2PAGE=OFF -DENABLE_REDIS=OFF -DENABLE_TESTS=OFF -DENABLE_SAMPLES=OFF -DENABLE_MSVC_MP=OFF
    )
    set(ENV{POCO_PATH} ${CMAKE_CURRENT_LIST_DIR}/_install)
    add_dependencies(hfetch POCO)
    #set(ALL_LIBS ${ALL_LIBS} POCO)
else()

    set(ALL_LIBS ${ALL_LIBS} ${POCO})
    set(ENV{POCO_PATH} ${POCO})
endif()

if (CASS)
    message("Using system c++ cassandra driver")
    set(ALL_LIBS ${ALL_LIBS}  ${CASS})
else()
    message("Downloading our own CASS DRIVER")
    unset(CASS)
    ExternalProject_Add(
            CASS
            URL https://github.com/datastax/cpp-driver/archive/2.5.0.tar.gz
            URL_HASH SHA1=70036c9d9685daf37ccec95851e1b1823357b4ec
            CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_LIST_DIR}/_install -DCASS_USE_LIBSSH2=OFF -DCASS_USE_OPENSSL=OFF
    )
    add_dependencies(hfetch CASS)
    # set(ALL_LIBS ${ALL_LIBS} CASS)


endif()

#
set(TBB_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/_install)
set(TBB_LIBRARY ${CMAKE_CURRENT_LIST_DIR}/_install/lib)
set(TBB_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/_install/include)
FIND_PACKAGE(TBB) # searches for the system TBB
#FIND_LIBRARY(TBB NAMES tbb PATHS ${CMAKE_CURRENT_LIST_DIR}/_install/lib)

#if (NOT TBB_FOUND)

if (NOT TBB_FOUND)
    message(Downloading our own TBB)
    unset(TBB)
    ExternalProject_Add(
            TBB
            URL "https://www.threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb2017_20161128oss_src.tgz"
            CMAKE_COMMAND echo
            BUILD_COMMAND make tbb
            BUILD_IN_SOURCE 1
            INSTALL_COMMAND cp -r build/linux_intel64_gcc_cc6.3.1_libc2.23_kernel4.9.8_release/libtbb.so build/linux_intel64_gcc_cc6.3.1_libc2.23_kernel4.9.8_release/libtbb.so.2  ${CMAKE_CURRENT_LIST_DIR}/_install/lib && cp -r include/tbb ${CMAKE_CURRENT_LIST_DIR}/_install/include
    )
    add_dependencies(hfetch TBB)    #set(ALL_LIBS ${ALL_LIBS} TBB)
    #set(ENV{TBB_PATH} ${CMAKE_CURRENT_LIST_DIR}/_install)
else()
    set(ALL_LIBS ${ALL_LIBS} ${TBB_LIBRARIES})
    set(ALL_INCLUDE ${ALL_INCLUDE} ${TBB_INCLUDE_DIRS})
endif()


LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/_install)
#Python setup

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS} ${CMAKE_CURRENT_LIST_DIR}/_install/include ${ALL_INCLUDE})

set(ALL_LIBS ${ALL_LIBS} ${PYTHON_LIBRARIES})


target_include_directories(hfetch PRIVATE ${CMAKE_CURRENT_LIST_DIR}/_install/include/ )
target_compile_definitions(hfetch PRIVATE ${TBB_DEFINITIONS})
set_target_properties(hfetch PROPERTIES PREFIX "")
target_link_libraries(hfetch ${ALL_LIBS})



FIND_LIBRARY(PT NAMES pthread PATHS /lib)

find_package(GTest)
include_directories(${GTEST_INCLUDE_DIRS})

enable_testing()
add_executable(hfetch_test tests/runtests.cpp)

target_link_libraries( hfetch_test ${ALL_LIBS} ${PT}  ${GTEST_BOTH_LIBRARIES} ${hfecth} )
add_test(test hfetch_test)
#End of dependencies


#install(CODE "execute_process(COMMAND python setup.py install -f WORKING_DIRECTORY ../)")




