cmake_minimum_required(VERSION 3.1)
project(tksvm-gauge CXX)

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
  TKSVM_SIMINCL=tksvm/gauge/gauge_sim.hpp
  )

set(GAUGE_SRC
  src/gauge_sim.cpp
  src/nematicities.cpp
  src/point_groups.cpp
  src/phase_diagram.cpp
  )

add_executable(gauge-sample ${TKSVM_SAMPLE_SRC} ${GAUGE_SRC})
add_executable(gauge-learn ${TKSVM_LEARN_SRC} ${GAUGE_SRC})
add_executable(gauge-test ${TKSVM_TEST_SRC} ${GAUGE_SRC})
add_executable(gauge-coeffs ${TKSVM_COEFFS_SRC} ${GAUGE_SRC})
add_executable(gauge-segregate-phases ${TKSVM_SEGREGATE_PHASES_SRC} ${GAUGE_SRC})

target_link_libraries(gauge-sample ${ALPSCore_LIBRARIES} ${TKSVM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(gauge-learn ${ALPSCore_LIBRARIES} ${TKSVM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(gauge-test ${ALPSCore_LIBRARIES} ${TKSVM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(gauge-coeffs ${ALPSCore_LIBRARIES} ${TKSVM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(gauge-segregate-phases ${ALPSCore_LIBRARIES} ${TKSVM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS
  gauge-sample
  gauge-learn
  gauge-test
  gauge-coeffs
  gauge-segregate-phases
  DESTINATION bin)
