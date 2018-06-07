[Probing Hidden Spin Order with Interpretable Machine Learning][1]
==================================================================

This is the source code accompanying the manuscript [arxiv:1804.08557][1].

Requirements
------------

These codes are based on the [ALPSCore][2] library. Refer to [their website][3]
for installation instructions. ALPSCore requires the Boost and HDF5 libraries.

Further, we require a modern C++ compiler (tested with GCC 7 or higher) with
C++14 support. For the solution of the SVM optimization problem, we rely on
the [libsvm][4] library which is included as part of our
self-developed [C++ wrapper library][5] which is kept _in-tree_ as a git
submodule and comes included in the tarball release. Lastly, [Eigen 3][8] is
used for linear algebra.

#### Regarding Boost and C++11

Boost will conditionally compile enums to use C++11's scoped enums when
available and otherwise fall back to type-unsafe enums. Since Boost headers are
used directly in our code and included indirected through ALPSCore headers, one
has be cautious of linker errors due to a mismatch of C++11 and C++03 symbols.
We recommend using a C++11 compiler throughout (GCC 6+ uses C++11 by default, or
provide the `-DCMAKE_CXX_FLAGS="-std=c++11"` flag to CMake) to build a local
copy of Boost, and compile ALPSCore and our codes. ALPSCore can be advised to use
the local version of Boost by providing the
`-DBOOST_ROOT=/path/to/boost/install` flag to CMake when building the ALPSCore
libraries.

### Regarding HDF5 1.10

There is a known upstream bug in HDF5 1.10 that persisted at least until version
1.10.1. We advise to use version 1.8.19 for the time being. If compatibility to
the 1.10-format is urgently required, 1.10 may be patched to resolve the bug.
Refer to ALPSCore [issue #290][6]
for details regarding this workaround.


Building and Installation
-------------------------

### Building and installing ALPSCore

Detailed instructions on [how to build ALPSCore][7] can be fournd in the
project's wiki. The procedure revolves around the following:

    $ cd alpscore
    $ mkdir build.tmp && cd build.tmp
    $ cmake ..
    $ make -jN
    $ make test
    $ make install
    
Replace `N` with the number of processors you want to use to build, e.g. `-j8`.
You may want to specify additional flags to `cmake`:

  * `-DCMAKE_INSTALL_PREFIX=$HOME/.local`, or another custom install location.
    This is required if you don't have permission to write at the default
    install prefix (`/usr/local`). Mind that ALPSCore installs a CMake script
    that has to be picked up by CMake when building our codes. Thus, any
    non-standard install location needs to be matched by a
    `-DCMAKE_PREFIX_PATH=<...>` flag when configuring the client code.
  * If a local version of boost has been install
    ([see above](#regarding-boost-and-c11)), point CMake to it by specifying
    `-DBOOST_ROOT=/path/to/boost/install`. Otherwise your local version may not
    be found, or be shadowed by an incompatible version.

### Building our client codes

Our codes also use CMake to configure the build environment. The procedure is
analogous to ALPSCore's, e.g.:

    $ cd svm-order-params
    $ mkdir build && cd build
    $ cmake ..
    $ make -jN all

Note that the repository makes use of git submodules. To also clone the
dependent repositories `svm` and `colormap`, supplement your call to `git clone`
with the flag `--recursive`; if the repository is already cloned, run

    $ git submodule update --init

Seven executables will be linked to *learn* an SVM model from Monte Carlo
data, *test* the resulting model to obtain an order parameter curve, and to
extract the *coeff*icient*s* for the Ising and orientational gauge model,
respectievly. In the following, we'll focus on the latter.

Additionally, the program `output-contractions` calculates and saves the "masks"
for every contraction at a given rank. These masks correspond to columns in the
contraction matrix that is used in the least-squares fit of the full coefficient
matrix.


Usage
-----

### Parameter files

Simulation parameters are stored in INI-style files. We provide example
parameter files for the high-rank symmetries discussed in the [paper][1]
(`Td.ini`, `Oh.ini`, `Th.ini`, and `Ih.ini`), as well as the quadrupolar order
(`Dinfh.ini`) and Heisenberg antiferromagnetism (`AFM.ini`).

### Learning

    $ ./gauge-learn Td.ini

will carry out Monte Carlo simulations at an ensemble of temperatures,
periodically sampling the configuration vector and storing the corresponding
temperature in an `svm::problem`. The latter is stored in a file `Td.clone.h5`.

**Random walk in temperature:** Each OpenMP thread will perform a Markov chain
random walk through temperature space. The resulting target distributed is a
Gaussian centered at `temp_crit` with standard deviation `temp_sigma`. At each
temperature, `thermalization_sweeps` Monte Carlo steps are carried out if the
temperature has changed, followed by another `total_sweeps` Monte Carlo steps
during which a total of `N_sample` snapshots of the configuration vector are
taken and stored in the `svm::problem` along with the current temperature.
Subsequently, an attempt will be made to change the temperature by an absolute
amount of at most `temp_step`, which may or may not be accepted according to
Markov chain Metropolis probabilities. This counts as one temperature step. A
total of `N_temp` temperature steps will be done.

**Time limit:** If the required amount of temperature steps cannot be carried
out within `timelimit` seconds, the simulation terminates prematurely and the
incomplete `svm::problem` is written to `Td.clone.h5`. One can resume the
simulation with

    $ ./gauge-learn Td.clone.h5
    
or skip ahead to the SVM optimization phase:

    $ ./gauge-learn Td.clone.h5 --skip-sampling

**SVM optimization:** The optimization problem will be solved once sampling
concluded. The training samples are classified into ordered and disordered phase
according to `temp_crit` just prior to that. `nu` is the regularization
parameter in ν-SVM. Note that `temp_crit` and `nu` may be
overriden via argument *after* sampling,

    $ ./gauge-learn Td.clone.h5 --skip-sampling --nu 0.4 --temp_crit 0.3

Note that changing `temp_crit` for classification only can lead to severely
imbalanced classification problems (many more samples in one class than the
other) which then necessitates a lower `nu` regularization in order for the
underlying optimization problem not to become infeasible, cf. Fig. 5 in
the [paper][1].

The result of the optimization is stored in `Td.out.h5`.

### Testing

    $ ./gauge-test Td.out.h5
    
will read the SVM model that was previously learnt and measure its decision
function as an observable in independent Monte Carlo simulations at
`test.N_temp` temperatures on an equidistant grid between `test.temp_min` and
`test.temp_max` to obtain the order parameter curve, (cf. [Fig. 1][1]). Results
are summarized in a text file `Td.test.txt` and full observables are stored in
`Td.test.h5`.

### Extracting coefficient matrix

    $ ./gauge-coeffs Td.out.h5

reads the SVM model and contracts over the support vectors to extract the
coefficient matrix (cf. our [manuscript][1]), block structure, and performs
miscellaneous analyses and processing steps, such as fitting and removing
self-contractions or comparing to the exact result.

The program has two major modes: extraction of the full coefficient matrix and
*single-block mode* where a single block is targetted and extracted exclusively.
This is done by specifying the command line parameter `--block=<block-spec>`
where `<block-spec>` identifies a block by its color indices. E.g. in the case
of the tetrahedral (rank-3) order, the non-trival block can be found by
specifying:

    --block=[lmn:lmn]

A number of flags can be used to customize the behavior of the program:

* `-v | --verbose`: print information on what happens.
* `-u | --unsymmetrize`: coefficients involving redundant monomial
  (cf. [Supplementary Materials][1]) are reconstructed, i.e. the coefficient
  value of the corresponding non-redundant coefficient is copied.
* `-r | --raw`: the indicies are *not* rearranged. By default, the indices are
  reshuffled in the form `(α_1, ..., α_n, a_1, ..., a_n)` and lexicographically
  ordered such that the color-block structure becomes apparent. Specifying this
  flag disables this, and indices are arranged as `(α_1, a_1, ..., α_n, a_n)`.
* `-c | --contraction-weights`: for each block (or only the single block),
  perform a least-squares fit of contraction "masks" to obtain the coefficients
  with which each contraction contributes and output those.
* `-s | --remove-self-contractions`: perform the same analysis as above, but
  consequently isolate the contributions due to self-contractions and subtract
  them from the full coefficient matrix.
* `-e | --exact`: also calculate and output the exact solution if availble
  (currently only for `Dinfh` and `Td`). Requires `--unsymmetrize`. Unavailable
  in single-block mode.
* `-d | --diff`: calculate and output the difference of the coefficient matrix to the
  exact solution if available. Requires `--unsymmetrize`. Unavailable in
  single-block mode.
* `-b | --blocks-only`: skip output of the full coefficient matrix and only output
  block structure. Unavailable (and pointless) in single-block mode.

Short flags may be combined into one multi-flag, as is customary for
POSIX-compatible programs.

Note that "blocks" refers to the representation where indices have been
reshuffled. When the full coefficient matrix is extracted, the contraction
analysis (`-c`, `-s`) operates on the symmetrized representation where redundant
element have not been reinstated and "blocks" are actually non-local. In
single-block mode however, this would not work because the equivalents of
redundant elements are actually part of different blocks. Thus, in single-block
mode, the contraction-analysis is performed on the "unsymmetrized" (redundant)
representation. Consequently, the contraction coefficients will be different
from those obtained for the same block in full-matrix mode, as now equivalent
contractions will contribute evenly, as opposed to only those contractions that
are compatible with the symmetrization.

The three panels in [Fig. 2 of the manuscript][1] can be reproduced by the
following invocations:

    $ ./gauge-coeffs -u Td.out.h5
    $ ./gauge-coeffs --block=[lmn:lmn] -u Td.out.h5
    $ ./gauge-coeffs --block=[lmn:lmn] -us Td.out.h5

License
-------

Copyright © 2018  Jonas Greitemann, Ke Liu, and Lode Pollet

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the GNU General Public License is available in the
file [LICENSE](LICENSE).


[1]: https://arxiv.org/abs/1804.08557
[2]: https://github.com/ALPSCore/ALPSCore
[3]: http://alpscore.org/
[4]: https://github.com/cjlin1/libsvm
[5]: https://github.com/jgreitemann/svm
[6]: https://github.com/ALPSCore/ALPSCore/issues/290
[7]: https://github.com/ALPSCore/ALPSCore/wiki/Installation
[8]: https://eigen.tuxfamily.org/
