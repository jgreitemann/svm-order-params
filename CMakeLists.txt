cmake_minimum_required(VERSION 2.8.12)
project(tksvm CXX)
set(CMAKE_CXX_STANDARD 14 PARENT_SCOPE)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(ALPSCore 2.2.0 REQUIRED)
find_package(Eigen3 REQUIRED)

option(SVM__ENABLE_TESTS "If tests for SVM submodule should be compiled" OFF)
add_subdirectory(svm)

set(CONFIG_MAPPING "EAGER" CACHE STRING "Perform mapping of spin configurations lazily or eagerly")
if(CONFIG_MAPPING STREQUAL LAZY)
  set(TKSVM_DEFINITIONS "${TKSVM_DEFINITIONS} CONFIG_MAPPING_LAZY" PARENT_SCOPE)
  message(STATUS "Using lazy mapping of spin configurations")
elseif(CONFIG_MAPPING STREQUAL EAGER)
  message(STATUS "Using eager mapping of spin configurations")
else()
  message(FATAL_ERROR "Unknown configuration mapping")
endif()

get_filename_component(BASEDIR . ABSOLUTE)

set(TKSVM_INCLUDE_DIRS
  ${BASEDIR}/include
  ${BASEDIR}/colormap/include
  ${BASEDIR}/svm/include
  PARENT_SCOPE)

set(TKSVM_SAMPLE_SRC
  ${BASEDIR}/src/sample.cpp
  PARENT_SCOPE)

set(TKSVM_LEARN_SRC
  ${BASEDIR}/src/learn.cpp
  PARENT_SCOPE)

set(TKSVM_TEST_SRC
  ${BASEDIR}/src/test.cpp
  PARENT_SCOPE)

set(TKSVM_COEFFS_SRC
  ${BASEDIR}/src/coeffs.cpp
  ${BASEDIR}/src/utilities/contraction.cpp
  ${BASEDIR}/src/utilities/results.cpp
  PARENT_SCOPE)

set(TKSVM_SEGREGATE_PHASES_SRC
  ${BASEDIR}/src/segregate_phases.cpp
  PARENT_SCOPE)

set(TKSVM_LIBRARIES
  svm
  PARENT_SCOPE)
