cmake_minimum_required(VERSION 2.8.12)
project(svm-order-params CXX)

enable_testing()

add_subdirectory(svm)
include_directories(svm)
include_directories(svm/libsvm)

include_directories(ising)

add_executable(svm-ising-learn main.cpp ising/ising.cpp svm/data_view.cpp)

add_executable(ising-coeffs ising-coeffs.cpp ising/ising.cpp svm/data_view.cpp)

# Request the ALPSCore package (with all components)
# The ALPSCore package must be installed in some
# standard place (like /usr/local),
# or somewhere in your PATH. 
# Otherwise, point ALPSCore_DIR environment variable
# to the ALPScore installation tree.
find_package(ALPSCore 1.0 REQUIRED)

# Use ALPSCore_LIBRARIES variable to link to ALPSCore 
target_link_libraries(svm-ising-learn ${ALPSCore_LIBRARIES} svm)
target_link_libraries(ising-coeffs ${ALPSCore_LIBRARIES} svm)
