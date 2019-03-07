#!/bin/bash

git submodule update --init

# Set working dir to script location
cd "${0%/*}"

# Build phase (~5 min)
mkdir -p build
pushd build
echo "Building code with D2h classifier"
cmake -DGAUGE_CLASSIFIER=D2h -DCMAKE_INSTALL_PREFIX=../D2h ../..
make install -j
echo "Building code with D3h classifier"
cmake -DGAUGE_CLASSIFIER=D3h -DCMAKE_INSTALL_PREFIX=../D3h ../..
make install -j
echo "Building code with hyperplane classifier"
cmake -DGAUGE_CLASSIFIER=HYPERPLANE -DCMAKE_INSTALL_PREFIX=../hyperplane ../..
make install -j
echo "Building code with fixed_from_cycle classifier"
cmake -DGAUGE_CLASSIFIER=CYCLE -DCMAKE_INSTALL_PREFIX=../cycle ../..
make install -j
echo "Building code with fixed_from_grid classifier"
cmake -DGAUGE_CLASSIFIER=GRID -DCMAKE_INSTALL_PREFIX=../grid ../..
make install -j
popd

echo "Ising model; binary classification based on crit. temperature"
pushd hyperplane/ising
../bin/ising-learn ising.ini # ~2 min
../bin/ising-coeffs ising.out.h5 # ~30 sec
../bin/ising-test ising.out.h5 --total_sweeps=100000 # ~6 min
popd

echo "Cinfv symmetry; binary classification based on crit. temperature"
pushd hyperplane/Cinfv
../bin/gauge-learn Cinfv.ini # ~6 min
../bin/gauge-coeffs Cinfv.out.h5 -u # < 1 sec
../bin/gauge-test Cinfv.out.h5 # ~3 min
popd

echo "Dinfv symmetry; binary classification based on crit. temperature"
pushd hyperplane/Dinfh
../bin/gauge-learn Dinfh.ini # ~6 min
../bin/gauge-coeffs Dinfh.out.h5 -u --diff # < 1 sec
../bin/gauge-test Dinfh.out.h5 # ~4 min
popd

echo "Td symmetry; binary classification based on crit. temperature"
pushd hyperplane/Td
../bin/gauge-learn Td.ini # ~4 min
../bin/gauge-coeffs Td.out.h5 -u --diff # ~5 sec
mkdir lmn_block
pushd lmn_block
../../bin/gauge-coeffs ../Td.out.h5 -u --block=[lmn:lmn] # < 1 sec
popd
mkdir lmn_block_without_sc
pushd lmn_block_without_sc
../../bin/gauge-coeffs ../Td.out.h5 -u --block=[lmn:lmn] --remove-self-contractions # < 1 sec
popd
../bin/gauge-test Td.out.h5 # ~5 min
popd

echo "Th symmetry; binary classification based on crit. temperature"
pushd hyperplane/Th
../bin/gauge-learn Th.ini # ~4 min
../bin/gauge-coeffs Th.out.h5 -u --blocks-only # ~1 min
../bin/gauge-test Th.out.h5 # ~6 min
popd

echo "Oh symmetry; binary classification based on crit. temperature"
pushd hyperplane/Oh
../bin/gauge-learn Oh.ini # ~5 min
../bin/gauge-coeffs Oh.out.h5 -u --blocks-only # ~1 min
../bin/gauge-test Oh.out.h5 # ~4 min
popd

echo "Td symmetry; training deep inside either phase"
pushd cycle/Td
../bin/gauge-learn Td.ini # ~4 min
../bin/gauge-coeffs Td.out.h5 -u # < 1 sec
../bin/gauge-test Td.out.h5 --total_sweeps=5000 # ~6 min
popd

echo "D2h symmetry; training at four corners of the phase diagram"
pushd cycle/D2h
../bin/gauge-learn D2h.ini # ~23 min
../bin/gauge-coeffs D2h.out.h5 -u # < 1 sec
popd

echo "D2h symmetry; training on a grid; classification according to phase diagram"
pushd D2h
bin/gauge-learn D2h.ini # ~29 min
bin/gauge-coeffs D2h.out.h5 -u # < 1 sec
bin/gauge-test D2h.out.h5 # ~11 min
popd

echo "D2h symmetry; training on a grid; classification according to grid points"
pushd grid
# Sample from scratch:
# bin/gauge-learn D2h.ini > /dev/null # ~35 min
# Or use existing samples from previous calculation:
ln -s ../D2h/D2h.clone.h5
bin/gauge-learn D2h.clone.h5 --skip-sampling --nu=0.1 > /dev/null # ~ 1 min
bin/gauge-segregate-phases D2h.out.h5 --rhoc=1.6 # < 1 sec
popd

echo "D3h symmetry (rank 2); training on a grid; classification according to phase diagram"
pushd D3h/rank_2
../bin/gauge-learn D3h.ini # ~33 min
../bin/gauge-coeffs D3h.out.h5 -u # < 1 sec
popd

echo "D3h symmetry (rank 3); training on a grid; classification according to phase diagram"
pushd D3h/rank_3
../bin/gauge-learn D3h.ini # ~34 min
../bin/gauge-coeffs D3h.out.h5 -u # ~10 sec
popd
