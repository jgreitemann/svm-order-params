cmake_minimum_required(VERSION 2.8.12)
project(frustmag CXX)

set(CMAKE_CXX_STANDARD 14)
option(USE_CONCEPTS "Use experimental C++20 concepts" ON)
if(USE_CONCEPTS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
  add_definitions(-DUSE_CONCEPTS)
endif()

find_package(ALPSCore 2.2.0 REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(.)
include_directories(..)
include_directories(../utilities)

enable_testing()
add_subdirectory(test)
