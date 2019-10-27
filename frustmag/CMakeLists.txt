cmake_minimum_required(VERSION 3.1)
project(tksvm-frustmag CXX)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

add_compile_options(
  "-Wall" "-Wextra"
  "-Wno-unknown-pragmas"
  "-Wno-c++17-extensions"
  "-fdiagnostics-color=always"
  )

set(HAMILTONIAN "heisenberg" CACHE STRING "The Hamiltonian for the frustmag simulation")
if(HAMILTONIAN STREQUAL heisenberg)
  add_definitions(-DHEISENBERG_HAMILTONIAN)
  message(STATUS "Using frustmag Hamiltonian: heisenberg")
elseif(HAMILTONIAN STREQUAL ising)
  add_definitions(-DISING_HAMILTONIAN)
  message(STATUS "Using frustmag Hamiltonian: ising")
else()
  message(FATAL_ERROR "Unknown frustmag Hamiltonian")
endif()

set(LATTICE "kagome" CACHE STRING "The lattice for the frustmag simulation")
if(LATTICE STREQUAL chain)
  add_definitions(-DCHAIN)
  message(STATUS "Using frustmag lattice: chain")
elseif(LATTICE STREQUAL square)
  add_definitions(-DSQUARE)
  message(STATUS "Using frustmag lattice: square")
elseif(LATTICE STREQUAL cubic)
  add_definitions(-DCUBIC)
  message(STATUS "Using frustmag lattice: cubic")
elseif(LATTICE STREQUAL triangular)
  add_definitions(-DTRIANGULAR)
  message(STATUS "Using frustmag lattice: triangular")
elseif(LATTICE STREQUAL honeycomb)
  add_definitions(-DHONEYCOMB)
  message(STATUS "Using frustmag lattice: honeycomb")
elseif(LATTICE STREQUAL kagome)
  add_definitions(-DKAGOME)
  message(STATUS "Using frustmag lattice: kagome")
elseif(LATTICE STREQUAL dice)
  add_definitions(-DDICE)
  message(STATUS "Using frustmag lattice: dice")
else()
  message(FATAL_ERROR "Unknown frustmag lattice")
endif()

find_package(ALPSCore 2.2.0 REQUIRED)
find_package(Threads)

add_subdirectory(.. tksvm)

include_directories(include ${TKSVM_INCLUDE_DIRS})
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

add_compile_definitions(
  ${TKSVM_DEFINITIONS}
  TKSVM_SIMINCL=tksvm/frustmag/config_frustmag_sim.hpp
  )

set(FRUSTMAG_SRC
  src/phase_diagram.cpp
  )

add_executable(frustmag-sample ${TKSVM_SAMPLE_SRC} ${FRUSTMAG_SRC})
add_executable(frustmag-learn ${TKSVM_LEARN_SRC} ${FRUSTMAG_SRC})
add_executable(frustmag-test ${TKSVM_TEST_SRC} ${FRUSTMAG_SRC})
add_executable(frustmag-coeffs ${TKSVM_COEFFS_SRC} ${FRUSTMAG_SRC})
add_executable(frustmag-segregate-phases ${TKSVM_SEGREGATE_PHASES_SRC} ${FRUSTMAG_SRC})

target_link_libraries(frustmag-sample ${ALPSCore_LIBRARIES} ${TKSVM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(frustmag-learn ${ALPSCore_LIBRARIES} ${TKSVM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(frustmag-test ${ALPSCore_LIBRARIES} ${TKSVM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(frustmag-coeffs ${ALPSCore_LIBRARIES} ${TKSVM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(frustmag-segregate-phases ${ALPSCore_LIBRARIES} ${TKSVM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS
  frustmag-sample
  frustmag-learn
  frustmag-test
  frustmag-coeffs
  frustmag-segregate-phases
  DESTINATION bin)
