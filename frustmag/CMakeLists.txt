cmake_minimum_required(VERSION 2.8.12)
project(frustmag CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(ALPSCore 2.2.0 REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(.)
include_directories(..)
include_directories(../utilities)

add_executable(heisenberg-chain
    main.cpp
    )
# Use ALPSCore_LIBRARIES variable to link to ALPSCore 
target_link_libraries(heisenberg-chain ${ALPSCore_LIBRARIES} stdc++fs)
