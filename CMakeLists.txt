cmake_minimum_required(VERSION 2.8.12)
project(svm-order-params CXX)

include_directories(ising)

add_executable(svm-mc-learn main.cpp ising/ising.cpp)

# Request the ALPSCore package (with all components)
# The ALPSCore package must be installed in some
# standard place (like /usr/local),
# or somewhere in your PATH. 
# Otherwise, point ALPSCore_DIR environment variable
# to the ALPScore installation tree.
find_package(ALPSCore 1.0 REQUIRED)

# Use ALPSCore_LIBRARIES variable to link to ALPSCore 
target_link_libraries(svm-mc-learn ${ALPSCore_LIBRARIES})
