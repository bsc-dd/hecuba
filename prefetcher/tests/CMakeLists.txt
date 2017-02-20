cmake_minimum_required(VERSION 3.3)
set(CMAKE_CXX_STANDARD 11)

project(tests)
include_directories( ${CMAKE_SOURCE_DIR}/_install/include /usr/include)
include(ExternalProject) # ExternalProject_Add

set(GTEST_ROOT $ENV{HOME}/local)

FIND_LIBRARY(PT NAMES pthread PATHS /lib)

find_package(GTest)
if (GTest_FOUND)
    include_directories(${GTEST_INCLUDE_DIRS})

    enable_testing()
    add_executable(hfetch_test runtests.cpp)

    target_link_libraries( hfetch_test ${ALL_LIBS} ${PT}  ${GTEST_BOTH_LIBRARIES} hfetch )
    add_test(test hfetch_test)
endif()

