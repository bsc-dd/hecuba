cmake_minimum_required(VERSION 3.3)
set(CMAKE_CXX_STANDARD 11)

IF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Debug) #Release|Debug|RelWithDebInfo|MinSizeRel
ENDIF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -fPIC")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")
#set(CMAKE_POSITION_INDEPENDENT_CODE YES)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(ENABLE_PRECOMPILED_HEADERS OFF)

include(ExternalProject) # ExternalProject_Add

message("CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

project(hfetch)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if (NOT $ENV{CPLUS_INCLUDE_PATH}  STREQUAL "")
    string (REPLACE ":" " " INCLUDES $ENV{CPLUS_INCLUDE_PATH})
    string (REPLACE "//" "/" INCLUDES ${INCLUDES})
    string(REGEX MATCHALL "[^ ]+" LIST_INCLUDES ${INCLUDES})
endif (NOT $ENV{CPLUS_INCLUDE_PATH}  STREQUAL "")
include_directories(LIST_INCLUDES)

if (NOT $ENV{CPATH}  STREQUAL "")
    string (REPLACE ":" " " INCLUDES $ENV{CPLUS_INCLUDE_PATH})
    string (REPLACE "//" "/" INCLUDES ${INCLUDES})
    string(REGEX MATCHALL "[^ ]+" LIST_INCLUDES ${INCLUDES})
endif (NOT $ENV{CPATH}  STREQUAL "")


include_directories(LIST_INCLUDES)
include_directories(${CMAKE_CURRENT_LIST_DIR}/_install/include)
set(SOURCE_FILES Prefetch.cpp HCache.cpp CacheTable.cpp TupleRow.cpp TupleRowFactory.cpp ModuleException.cpp Writer.cpp )
set(HEADER_FILES Prefetch.h HCache.h CacheTable.h TupleRow.h TupleRowFactory.h ModuleException.h Writer.h metadata.h)



if (NOT $ENV{LD_LIBRARY_PATH}  STREQUAL "")
    string (REPLACE ":" " " ENV_LIBS $ENV{LD_LIBRARY_PATH})
    string (REPLACE "//" "/" ENV_LIBS ${ENV_LIBS})
    string(REGEX MATCHALL "[^ ]+" LIST_LIBS ${ENV_LIBS})
endif (NOT $ENV{LD_LIBRARY_PATH}  STREQUAL "")

LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/_install/lib ${LIST_LIBS})

add_library(hfetch SHARED ${SOURCE_FILES} ${HEADER_FILES})



FIND_LIBRARY(POCO NAMES PocoFoundation PATHS ${CMAKE_CURRENT_LIST_DIR}/_install/lib)
FIND_LIBRARY(CASS NAMES cassandra PATHS ${CMAKE_CURRENT_LIST_DIR}/_install/lib)
FIND_LIBRARY(LIBUV NAMES uv PATHS ${CMAKE_CURRENT_LIST_DIR}/_install/lib)


IF (NOT LIBUV)
    message("Downloading LIBUV requisite for Cassandra C++ driver " ${LIBUV})
    unset(LIBUV)
    ExternalProject_Add(
            LIBUV
            URL "https://github.com/libuv/libuv/archive/v1.11.0.tar.gz"
            URL_HASH SHA1=54f0972aa0d3f6a6036d477b381c01f030f9a2b5
            CMAKE_COMMAND echo
            BUILD_COMMAND sh autogen.sh COMMAND ./configure --prefix=${CMAKE_CURRENT_LIST_DIR}/_install
            BUILD_IN_SOURCE 1
            INSTALL_COMMAND make COMMAND make install
    )
    add_dependencies(hfetch LIBUV)
else()
    message("Using system's LIBUV: " ${LIBUV})
    set(ALL_LIBS ${ALL_LIBS} ${LIBUV})
endif()


if(NOT POCO)
    message("Downloading Poco Libs")
    unset(POCO)
    ExternalProject_Add(
            POCO
            URL https://pocoproject.org/releases/poco-1.7.7/poco-1.7.7.tar.gz
            URL_HASH SHA1=bdac2ff5cb2bea078cc71f1d85704bbeba7e14c9
            CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_LIST_DIR}/_install -DENABLE_CPPUNIT=OFF -DENABLE_XML=OFF -DENABLE_JSON=OFF -DENABLE_MONGODB=OFF -DENABLE_PDF=OFF -DENABLE_UTIL=OFF -DENABLE_NET=OFF -DENABLE_NETSSL=OFF -DENABLE_NETSSL_WIN=OFF -DENABLE_CRYPTO=OFF -DENABLE_DATA=OFF -DENABLE_DATA_SQLITE=OFF -DENABLE_DATA_MYSQL=OFF -DENABLE_DATA_POSTGRESQL=OFF -DENABLE_DATA_ODBC=OFF -DENABLE_SEVENZIP=OFF -DENABLE_ZIP=OFF -DENABLE_APACHECONNECTOR=OFF -DENABLE_CPPPARSER=OFF -DENABLE_POCODOC=OFF -DENABLE_PAGECOMPILER=OFF -DENABLE_PAGECOMPILER_FILE2PAGE=OFF -DENABLE_REDIS=OFF -DENABLE_TESTS=OFF -DENABLE_SAMPLES=OFF -DENABLE_MSVC_MP=OFF
    )
    add_dependencies(hfetch POCO)
else()
    message("Using system's POCO: " ${POCO})
    set(ALL_LIBS ${ALL_LIBS} ${POCO})
endif()



if (NOT CASS)
    message("Downloading C++ Cassandra Driver")
    unset(CASS)
    ExternalProject_Add(
            CASS
            DEPENDS LIBUV
            INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}/_install
            URL https://github.com/datastax/cpp-driver/archive/2.5.0.tar.gz
            URL_HASH SHA1=70036c9d9685daf37ccec95851e1b1823357b4ec
            CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_LIST_DIR}/_install -DCASS_USE_LIBSSH2=OFF -DCASS_USE_OPENSSL=OFF
    )
    add_dependencies(hfetch CASS)
    # set(ALL_LIBS ${ALL_LIBS} CASS)

else()

    message("Using system's C++ Cassandra driver: " ${CASS})
    set(ALL_LIBS ${ALL_LIBS}  ${CASS})
endif()





# FIND PACKAGES


set(TBB_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/_install)
set(TBB_LIBRARY ${CMAKE_CURRENT_LIST_DIR}/_install/lib)
set(TBB_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/_install/include)
FIND_PACKAGE(TBB) # searches for the system TBB


if (NOT TBB_FOUND)
    message(Downloading Intel TBB)
    unset(TBB)
        #set(ENV{tbb_build_prefix} tbb_pre)
        set(build_prefix tbb_prefix)
    ExternalProject_Add(
            TBB
            URL "https://www.threadingbuildingblocks.org/sites/default/files/software_releases/source/tbb2017_20161128oss_src.tgz"
            CMAKE_COMMAND echo
            BUILD_COMMAND make tbb tbb_build_prefix=${build_prefix}
            BUILD_IN_SOURCE 1
            INSTALL_COMMAND mkdir -p  ${CMAKE_CURRENT_LIST_DIR}/_install/lib
            COMMAND  mkdir -p ${CMAKE_CURRENT_LIST_DIR}/_install/include
            COMMAND cp -r build/${build_prefix}_release/libtbb.so build/${build_prefix}_release/libtbb.so.2  ${CMAKE_CURRENT_LIST_DIR}/_install/lib
            COMMAND cp -r include/tbb ${CMAKE_CURRENT_LIST_DIR}/_install/include
    )
    add_dependencies(hfetch TBB)
    #set(ALL_LIBS ${ALL_LIBS} "libtbb.so")
else()
    set(ALL_LIBS ${ALL_LIBS} ${TBB_LIBRARIES})
    include_directories(${TBB_INCLUDE_DIRS})

endif()



find_package(PythonInterp 2.7)
find_package(PythonLibs 2.7.6 REQUIRED)

find_package(NumPy 1.8)
if (NUMPY_FOUND)
    include_directories(${NUMPY_INCLUDE_DIR})
else()
    message("Numpy headers not found, Numpy arrays support won't be available")
endif()



include_directories(${PYTHON_INCLUDE_DIRS})
set(ALL_LIBS ${ALL_LIBS} ${PYTHON_LIBRARIES})



set_target_properties(hfetch PROPERTIES PREFIX "")
target_link_libraries(hfetch ${ALL_LIBS})



FIND_LIBRARY(PT NAMES pthread PATHS /lib)

if (PT)
    set(GTEST_ROOT $ENV{HOME}/local)

    find_package(GTest)
    if (GTest_FOUND)
        include_directories(${GTEST_INCLUDE_DIRS})

        enable_testing()
        add_executable(hfetch_test tests/runtests.cpp)

        target_link_libraries( hfetch_test ${ALL_LIBS} ${PT}  ${GTEST_BOTH_LIBRARIES} hfetch )
        add_test(test hfetch_test)
    endif(GTest_FOUND)
endif(PT)