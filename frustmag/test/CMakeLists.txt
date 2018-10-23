include_directories(doctest)

add_executable(lattice lattice.cpp)
target_link_libraries(lattice ${ALPSCore_LIBRARIES} stdc++fs)
add_test(lattice lattice)
