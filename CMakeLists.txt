cmake_minimum_required(VERSION 2.8.12)
project(svm-order-params CXX)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

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

set(CONFIG_MAPPING "EAGER" CACHE STRING "Perform mapping of spin configurations lazily or eagerly")
if(CONFIG_MAPPING STREQUAL LAZY)
  add_definitions(-DCONFIG_MAPPING_LAZY)
  message(STATUS "Using lazy mapping of spin configurations")
elseif(CONFIG_MAPPING STREQUAL EAGER)
  message(STATUS "Using eager mapping of spin configurations")
else()
  message(FATAL_ERROR "Unknown configuration mapping")
endif()

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


include_directories(colormap)

include_directories(ising)
add_executable(ising-learn
    learn.cpp
    phase_space_policy.cpp
    checkpointing_stop_callback.cpp
    ising/ising.cpp
    )
target_compile_definitions(ising-learn PUBLIC -DSVMOP_SIMINCL=ising.hpp)

add_executable(ising-test
    test.cpp
    phase_space_policy.cpp
    ising/ising.cpp
    )
target_compile_definitions(ising-test PUBLIC -DSVMOP_SIMINCL=ising.hpp)

add_executable(ising-coeffs
    coeffs.cpp
    phase_space_policy.cpp
    ising/ising.cpp
    contraction.cpp
    results.cpp
    )
target_compile_definitions(ising-coeffs PUBLIC -DSVMOP_SIMINCL=ising.hpp)

include_directories(gauge)
add_executable(gauge-sample
    sample.cpp
    phase_space_policy.cpp
    gauge/gauge.cpp
    gauge/nematicities.cpp
    gauge/point_groups.cpp
    )
target_compile_definitions(gauge-sample PUBLIC -DSVMOP_SIMINCL=gauge.hpp)

add_executable(gauge-learn
    learn.cpp
    phase_space_policy.cpp
    checkpointing_stop_callback.cpp
    gauge/gauge.cpp
    gauge/nematicities.cpp
    gauge/point_groups.cpp
    )
target_compile_definitions(gauge-learn PUBLIC -DSVMOP_SIMINCL=gauge.hpp)

add_executable(gauge-test
    test.cpp
    phase_space_policy.cpp
    gauge/gauge.cpp
    gauge/nematicities.cpp
    gauge/point_groups.cpp
    )
target_compile_definitions(gauge-test PUBLIC -DSVMOP_SIMINCL=gauge.hpp)

add_executable(gauge-coeffs
    coeffs.cpp
    phase_space_policy.cpp
    gauge/gauge.cpp
    gauge/nematicities.cpp
    gauge/point_groups.cpp
    contraction.cpp
    results.cpp
    )
target_compile_definitions(gauge-coeffs PUBLIC -DSVMOP_SIMINCL=gauge.hpp)

add_executable(gauge-segregate-phases
    segregate_phases.cpp
    phase_space_policy.cpp
    gauge/gauge.cpp
    gauge/nematicities.cpp
    gauge/point_groups.cpp
    )
target_compile_definitions(gauge-segregate-phases PUBLIC -DSVMOP_SIMINCL=gauge.hpp)

add_executable(output-contractions
    output_contractions.cpp
    contraction.cpp
    )

include_directories(frustmag)

add_executable(frustmag-sample
    sample.cpp
    phase_space_policy.cpp
    )
target_compile_definitions(frustmag-sample PUBLIC -DSVMOP_SIMINCL=config_frustmag_sim.hpp -DHAS_SVM)

add_executable(frustmag-learn
    learn.cpp
    phase_space_policy.cpp
    )
target_compile_definitions(frustmag-learn PUBLIC -DSVMOP_SIMINCL=config_frustmag_sim.hpp -DHAS_SVM)

add_executable(frustmag-test
    test.cpp
    phase_space_policy.cpp
    )
target_compile_definitions(frustmag-test PUBLIC -DSVMOP_SIMINCL=config_frustmag_sim.hpp -DHAS_SVM)

add_executable(frustmag-coeffs
    coeffs.cpp
    phase_space_policy.cpp
    contraction.cpp
    results.cpp
    )
target_compile_definitions(frustmag-coeffs PUBLIC -DSVMOP_SIMINCL=config_frustmag_sim.hpp -DHAS_SVM)

add_executable(frustmag-segregate-phases
    segregate_phases.cpp
    phase_space_policy.cpp
    )
target_compile_definitions(frustmag-segregate-phases PUBLIC -DSVMOP_SIMINCL=config_frustmag_sim.hpp -DHAS_SVM)

# Use ALPSCore_LIBRARIES variable to link to ALPSCore
target_link_libraries(ising-learn ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(ising-test ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(ising-coeffs ${ALPSCore_LIBRARIES} svm stdc++fs)

target_link_libraries(gauge-sample ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(gauge-learn ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(gauge-test ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(gauge-coeffs ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(gauge-segregate-phases ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(output-contractions ${ALPSCore_LIBRARIES} svm stdc++fs)

target_link_libraries(frustmag-sample ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(frustmag-learn ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(frustmag-test ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(frustmag-coeffs ${ALPSCore_LIBRARIES} svm stdc++fs)
target_link_libraries(frustmag-segregate-phases ${ALPSCore_LIBRARIES} svm stdc++fs)

install(TARGETS
  ising-learn
  ising-test
  ising-coeffs
  gauge-sample
  gauge-learn
  gauge-test
  gauge-coeffs
  gauge-segregate-phases
  output-contractions
  frustmag-sample
  frustmag-learn
  frustmag-test
  frustmag-coeffs
  frustmag-segregate-phases
  DESTINATION bin)
