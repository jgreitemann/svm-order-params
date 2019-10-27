cmake_minimum_required(VERSION 3.1)
project(tksvm-ising CXX)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

add_compile_options(
  "-Wall" "-Wextra"
  "-Wno-unknown-pragmas"
  "-fdiagnostics-color=always"
  )

find_package(ALPSCore 2.2.0 REQUIRED)
find_package(Threads)

add_subdirectory(.. tksvm)

include_directories(include ${TKSVM_INCLUDE_DIRS})
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

add_compile_definitions(
  ${TKSVM_DEFINITIONS}
  TKSVM_SIMINCL=tksvm/ising/ising_sim.hpp
  )

set(ISING_SRC
  src/ising_sim.cpp
  )

add_executable(ising-sample ${TKSVM_SAMPLE_SRC} ${ISING_SRC})
add_executable(ising-learn ${TKSVM_LEARN_SRC} ${ISING_SRC})
add_executable(ising-test ${TKSVM_TEST_SRC} ${ISING_SRC})
add_executable(ising-coeffs ${TKSVM_COEFFS_SRC} ${ISING_SRC})

target_link_libraries(ising-sample ${ALPSCore_LIBRARIES} ${TKSVM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(ising-learn ${ALPSCore_LIBRARIES} ${TKSVM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(ising-test ${ALPSCore_LIBRARIES} ${TKSVM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(ising-coeffs ${ALPSCore_LIBRARIES} ${TKSVM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS
  ising-sample
  ising-learn
  ising-test
  ising-coeffs
  DESTINATION bin)
