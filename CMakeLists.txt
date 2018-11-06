cmake_minimum_required(VERSION 2.8.12)
project(svm-order-params CXX)

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
set(CMAKE_CXX_STANDARD 14)

find_package(ALPSCore 2.2.0 REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(.)
include_directories(utilities)

option(SVM__ENABLE_TESTS "If tests for SVM submodule should be compiled" OFF)
add_subdirectory(svm)

set(GAUGE_CLASSIFIER "D2h" CACHE STRING "The classifier used to label samples in the gauge sim")
if(GAUGE_CLASSIFIER STREQUAL HYPERPLANE)
  add_definitions(-DGAUGE_CLASSIFIER_HYPERPLANE)
  message(STATUS "Using gauge classifier: hyperplane")
elseif(GAUGE_CLASSIFIER STREQUAL CYCLE)
  add_definitions(-DGAUGE_CLASSIFIER_CYCLE)
  message(STATUS "Using gauge classifier: fixed_from_cycle")
elseif(GAUGE_CLASSIFIER STREQUAL D2h)
  add_definitions(-DGAUGE_CLASSIFIER_D2H)
  message(STATUS "Using gauge classifier: D2h_phase_diagram")
elseif(GAUGE_CLASSIFIER STREQUAL D3h)
  add_definitions(-DGAUGE_CLASSIFIER_D3H)
  message(STATUS "Using gauge classifier: D3h_phase_diagram")
elseif(GAUGE_CLASSIFIER STREQUAL GRID)
  add_definitions(-DGAUGE_CLASSIFIER_GRID)
  message(STATUS "Using gauge classifier: fixed_from_grid")
else()
  message(FATAL_ERROR "Unknown gauge classifier")
endif()

include_directories(colormap)

include_directories(ising)
add_executable(ising-learn
    learn.cpp
    phase_space_policy.cpp
    checkpointing_stop_callback.cpp
    ising/ising.cpp
    )
target_compile_definitions(ising-learn PUBLIC -DISING)

add_executable(ising-test
    test.cpp
    ising/ising.cpp
    )
target_compile_definitions(ising-test PUBLIC -DISING)

add_executable(ising-coeffs
    coeffs.cpp
    ising/ising.cpp
    contraction.cpp
    results.cpp
    )
target_compile_definitions(ising-coeffs PUBLIC -DISING)

include_directories(gauge)
add_executable(gauge-learn
    learn.cpp
    phase_space_policy.cpp
    checkpointing_stop_callback.cpp
    gauge/gauge.cpp
    gauge/nematicities.cpp
    gauge/point_groups.cpp
    )
target_compile_definitions(gauge-learn PUBLIC -DGAUGE)

add_executable(gauge-test
    test.cpp
    gauge/gauge.cpp
    gauge/nematicities.cpp
    gauge/point_groups.cpp
    )
target_compile_definitions(gauge-test PUBLIC -DGAUGE)

add_executable(gauge-coeffs
    coeffs.cpp
    gauge/gauge.cpp
    gauge/nematicities.cpp
    gauge/point_groups.cpp
    contraction.cpp
    results.cpp
    )
target_compile_definitions(gauge-coeffs PUBLIC -DGAUGE)

add_executable(gauge-segregate-phases
    segregate_phases.cpp
    phase_space_policy.cpp
    gauge/gauge.cpp
    gauge/nematicities.cpp
    gauge/point_groups.cpp
    )

target_compile_definitions(gauge-segregate-phases PUBLIC -DGAUGE)

add_executable(output-contractions
    output_contractions.cpp
    contraction.cpp
    )

# Use ALPSCore_LIBRARIES variable to link to ALPSCore 
target_link_libraries(ising-learn ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(ising-test ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(ising-coeffs ${ALPSCore_LIBRARIES} svm stdc++fs)

target_link_libraries(gauge-learn ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(gauge-test ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(gauge-coeffs ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(gauge-segregate-phases ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(output-contractions ${ALPSCore_LIBRARIES} svm stdc++fs)
