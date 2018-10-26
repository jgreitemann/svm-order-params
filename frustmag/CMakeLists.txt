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

add_executable(heisenberg-chain main.cpp)
target_compile_definitions(heisenberg-chain PUBLIC -DHEISENBERG -DCHAIN)

add_executable(ising-chain main.cpp)
target_compile_definitions(ising-chain PUBLIC -DISING -DCHAIN)

add_executable(heisenberg-square main.cpp)
target_compile_definitions(heisenberg-square PUBLIC -DHEISENBERG -DSQUARE)

add_executable(ising-square main.cpp)
target_compile_definitions(ising-square PUBLIC -DISING -DSQUARE)

add_executable(heisenberg-cubic main.cpp)
target_compile_definitions(heisenberg-cubic PUBLIC -DHEISENBERG -DCUBIC)

add_executable(ising-cubic main.cpp)
target_compile_definitions(ising-cubic PUBLIC -DISING -DCUBIC)

add_executable(heisenberg-triangular main.cpp)
target_compile_definitions(heisenberg-triangular PUBLIC -DHEISENBERG -DTRIANGULAR)

add_executable(ising-triangular main.cpp)
target_compile_definitions(ising-triangular PUBLIC -DISING -DTRIANGULAR)

# Use ALPSCore_LIBRARIES variable to link to ALPSCore 
target_link_libraries(heisenberg-chain ${ALPSCore_LIBRARIES} stdc++fs)
target_link_libraries(ising-chain ${ALPSCore_LIBRARIES} stdc++fs)
target_link_libraries(heisenberg-square ${ALPSCore_LIBRARIES} stdc++fs)
target_link_libraries(ising-square ${ALPSCore_LIBRARIES} stdc++fs)
target_link_libraries(heisenberg-cubic ${ALPSCore_LIBRARIES} stdc++fs)
target_link_libraries(ising-cubic ${ALPSCore_LIBRARIES} stdc++fs)
target_link_libraries(heisenberg-triangular ${ALPSCore_LIBRARIES} stdc++fs)
target_link_libraries(ising-triangular ${ALPSCore_LIBRARIES} stdc++fs)

enable_testing()
add_subdirectory(test)
