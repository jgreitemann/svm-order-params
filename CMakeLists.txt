cmake_minimum_required(VERSION 2.8.12)
project(svm-order-params CXX)

enable_testing()

find_package(ALPSCore 1.0 REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(svm)
include_directories(svm)
include_directories(svm/libsvm)

include_directories(ising)
add_executable(ising-learn
    learn.cpp
    ising/ising.cpp
    svm/data_view.cpp)
target_compile_definitions(ising-learn PUBLIC -DISING)

include_directories(gauge)
add_executable(gauge-learn
    learn.cpp
    gauge/gauge.cpp
    gauge/nematicities.cpp
    gauge/point_groups.cpp
    svm/data_view.cpp)
target_compile_definitions(gauge-learn PUBLIC -DGAUGE)

# Use ALPSCore_LIBRARIES variable to link to ALPSCore 
target_link_libraries(ising-learn ${ALPSCore_LIBRARIES} svm)
target_link_libraries(gauge-learn ${ALPSCore_LIBRARIES} svm)
