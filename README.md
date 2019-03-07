Probing hidden spin order with interpretable machine learning
=============================================================

These are the source codes accompanying the our papers:
* J. Greitemann, K. Liu, and Lode Pollet: _Probing hidden spin order with
interpretable machine learning_, [**Phys. Rev. B 99, 060404(R) (2019)**][9],
open access via [arXiv:1804.08557][1];
* K. Liu, J. Greitemann, and Lode Pollet: _Learning multiple order parameters
with interpretable machines_, [**Phys. Rev. B 99, 104410 (2019)**][11], open
access via [arXiv:1810.05538][10].

_Note: This version includes features which have been developed for the second
paper, specifically multiclassification. While this version is still capable
of producing the results presented in the first paper, you may want to check
out the version of the code that was originally published along with the first
paper. The relevant commit is tagged `PRB-99-060404(R)`, i.e. run_

```bash
$ git checkout PRB-99-060404(R)
```

Contents
--------

* [Dependencies](#dependencies)
* [Project structure](#project-structure)
* [Building and installation](#building-and-installation)
  - [Building and installing ALPSCore](#building-and-installing-alpscore)
  - [Building our client codes](#building-our-client-codes)
    - [Build configuration](#build-configuration)
* [Basic usage](#basic-usage)
  - [Parameter files](#parameter-files)
  - [Learning: sampling and SVM optimization](#learning-sampling-and-svm-optimization)
  - [Testing: measuring decision function and observables](#testing-measuring-decision-function-and-observables)
  - [Extracting the coefficient matrix](#extracting-the-coefficient-matrix)
  - [Spectral graph partitioning analysis](#spectral-graph-partitioning-analysis)
* [Runtime parameters](#runtime-parameters)
  - [Simulation runtime](#simulation-runtime)
  - [Phase diagram point specification](#phase-diagram-point-specification)
  - [SVM optimization (learning)](#svm-optimization-learning)
  - [Testing stage](#testing-stage)
  - [Client code-specific parameters](#client-code-specific-parameters)
  - [Sweep through the phase diagram](#sweep-through-the-phase-diagram)
  - [Classifiers](#classifiers)
* [Client code API](#client-code-api)
* [Changelog](#changelog)
* [License](#license)

Dependencies
------------

These codes are based on the [ALPSCore][2] library. Refer to [their website][3]
for installation instructions. ALPSCore requires the Boost and HDF5 libraries.

Further, we require a modern C++ compiler (tested with GCC 7 or higher) with
C++14 support. For the solution of the SVM optimization problem, we rely on the
[libsvm][4] library which is included as part of our self-developed [C++ wrapper
library][5] which is kept _in-tree_ as a git submodule. [Argh!][12] is used for
parsing command line options. Lastly, [Eigen 3][8] is used for linear algebra.

#### Regarding Boost and C++11

Boost will conditionally compile enums to use C++11's scoped enums when
available and otherwise fall back to type-unsafe enums. Since Boost headers are
used directly in our code and included indirected through ALPSCore headers, one
has be cautious of linker errors due to a mismatch of C++11 and C++03 symbols.
We recommend using a C++11 compiler throughout (GCC 6+ uses C++11 by default, or
provide the `-DCMAKE_CXX_FLAGS="-std=c++11"` flag to CMake) to build a local
copy of Boost, and compile ALPSCore and our codes. ALPSCore can be advised to
use the local version of Boost by providing the
`-DBOOST_ROOT=/path/to/boost/install` flag to CMake when building the ALPSCore
libraries.

Project structure
-----------------

This repository contains two client codes which are for the most part regular
Monte Carlo simulation codes based on the ALPSCore framework. In addition they
implement the relevant [API functions](#client-code-api) required by the SVM
framework.

The client codes are kept in their eponymous subdirectories:
* `ising`: This is a copy of the ALPSCore demo code for the 2D Ising model. It
  may be used to reproduce the results presented in [**P. Ponte and R.G. Melko,
  Phys. Rev. B 96, 205146 (2017)**][13].
* `gauge`: Lattice gauge model proposed in [**K. Liu, J. Nissinen, R.-J. Slager,
  K. Wu, and J. Zaanen, PRX 6, 041025 (2016)**][14] which lets us realize
  arbitrary orientational orders and was used to produce the result in our
  papers.

The project is designed as a generic framework that works with existing Monte
Carlo codes with minimal adaptation. Any code outside the client-code
directories is designed to be agnostic with respect to the client codes. For
each of the client codes, several executables are generated: `ising-learn`,
`ising-test`, `ising-coeffs`; `gauge-learn`, `gauge-test`, `gauge-coeffs`, and
`gauge-segregate-phases`:
* `*-learn`: Run the Monte Carlo simulations at different points in the phase
  diagram, occasionally sampling the spin configuration and applying the
  monomial mapping. The point in phase diagram space is recorded along with the
  mapped configuration. Once sampling concludes, the SVM optimization is
  performed and the resultant SVM model is saved to disk.
* `*-test`: Run the Monte Carlo simulation along a given line through the phase
  diagram, measuring the decision functions from the SVM optimization as
  observables (along with any other predefined observables for comparision).
* `*-coeffs`: Extract the coefficient matrix from the SVM model to infer the
  analytical order parameter.
* `*-segregate-phases`: Graph analysis to find the phase diagram without prior
  knowledge of the correct phase classification, cf. Sec. VII of
  [arXiv:1810.05538][10].


Building and installation
-------------------------

### Building and installing ALPSCore

Detailed instructions on [how to build ALPSCore][7] can be fournd in the
project's wiki. The procedure revolves around the following:

```bash
$ cd alpscore
$ mkdir build.tmp && cd build.tmp
$ cmake ..
$ make -jN
$ make test
$ make install
```

Replace `N` with the number of threads you want to use to build, e.g. `-j8`.
You may want to specify additional flags to `cmake`:

  * `-DCMAKE_INSTALL_PREFIX=$HOME/.local`, or another custom install location.
    This is required if you don't have permission to write at the default
    install prefix (`/usr/local`). Mind that ALPSCore installs a CMake script
    that has to be picked up by CMake when building our codes. Thus, any
    non-standard install location needs to be matched by a
    `-DCMAKE_PREFIX_PATH=<...>` flag when configuring the client code.
  * If a local version of boost has been installed
    ([see above](#regarding-boost-and-c11)), point CMake to it by specifying
    `-DBOOST_ROOT=/path/to/boost/install`. Otherwise your local version may not
    be found, or be shadowed by an incompatible version.

### Building our client codes

Our codes also use CMake to configure the build environment. The procedure is
analogous to ALPSCore's, e.g.:

```bash
$ cd svm-order-params
$ mkdir build && cd build
$ cmake ..
$ make -jN
```

Note that the repository makes use of git submodules. To also clone the
dependent repositories `svm` and `colormap`, supplement your call to `git clone`
with the flag `--recursive`; if the repository is already cloned, run

```bash
$ git submodule update --init
```

Finally, using `make install` the compiled executables can be copied to the
`bin` directory at the location configured in `CMAKE_INSTALL_PREFIX`. This step
is optional. The remainder of this README assumes that executables have been
installed to a directory in the user's `$PATH`.

#### Build configuration

Most of the behavior of the programs can be controlled through runtime
parameters. However, some options have to be selected at compile time via the
CMake variables listed below. To customize a variable, pass the appropriate flag
on to CMake, e.g.

```bash
$ cmake -DGAUGE_CLASSIFIER=CYCLE ..
```

or use the interactive `ccmake` configurator.

| Variable name          | Possible values                                                    |
|:-----------------------|:-------------------------------------------------------------------|
| `ALPSCore_DIR`         | `nonstandard/path/to/share/ALPSCore`                               |
| `Eigen3_DIR`           | `nonstandard/path/to/share/eigen3/cmake`                           |
| `CMAKE_BUILD_TYPE`     | `Release` (_default_), `Debug`                                     |
| `CMAKE_INSTALL_PREFIX` | path to install directory (executables will be copied into `bin/`) |
| `SVM__ENABLE_TESTS`    | `OFF` (_default_), `ON`                                            |
| `GAUGE_CLASSIFIER`     | `D2h` (_default_), `D3h`, `HYPERPLANE`, `CYCLE`, `GRID`            |


Basic usage
-----------

### Parameter files

Simulation parameters are stored in INI-style files.

We provide example parameter files for each client code, in the
[`params`](params) directory. Each parameter file comes with a brief comment
which explains its purpose, gives the build options (particularly the choice of
classifier that it is intended to be used with), and includes the shell commands
to run that particular simulation.

To build and run all the examples provided, execute the shell script:

```bash
$ params/build_and_run_all.sh
```

The results will be saved in the same directories as the parameter files. The
shell script also contains estimates for the run time of the individual steps
which took around 5 - 25 mins on an 8-core workstation. The full suite of
examples took between two and three hours.

In the following subsections, we will give a more detailed description of some
aspects of the learning, testing, and analysis stages. For a (somewhat)
comprehensive listing of supported parameters, refer to the section entitled
[_Runtime parameters_](#runtime-parameters).

### Learning: sampling and SVM optimization

**Sampling:** The `*-learn` binaries run the Monte Carlo simulation at a number
of points in the phase diagram. The configurations are sampled periodically and
stored as the _optimization problem_ in a file ending in `.clone.h5`. For
example,

```bash
$ gauge-learn Td.ini
```

will carry out Monte Carlo simulations of the gauge model with the tetrahedral
_T<sub>d</sub>_ symmetry group.

**Time limit:** If the required amount of Monte Carlo steps cannot be carried
out within `timelimit` seconds, the simulation terminates prematurely and the
incomplete data are written to `Td.clone.h5`. One can resume the simulation with

```bash
$ gauge-learn Td.clone.h5
```

or skip ahead to the SVM optimization phase:

```bash
$ gauge-learn Td.clone.h5 --skip-sampling
```

**Labeling samples:** Once the desired number of samples is collected, each
sample is labeled and the labeled samples are fed to the SVM. The labeling is
done by a component called the [classifier](#classifiers). The classifier has to
be selected at compile time using the `GAUGE_CLASSIFIER` build option. The
parameter file in [`params/hyperplane/Td/Td.ini`](params/D2h/D2h.ini) is
intended to by used with the `hyperplane` classifier (see below), mapping the
phase diagram points to a binary label, indicating if samples are from the
ordered or disordered phase, respectively.

**SVM optimization:** The SVM then proceeds to solve the classification problem.
The result of the optimization is stored in a file ending in `.out.h5` and to be
used as input for the consequent analysis.

**Multiclassification:** In contrast, when using the _D<sub>2h</sub>_ symmetry
group, the phase diagram features three phases called _O(3)_ (isotropic),
_D<sub>∞h</sub>_ (uniaxial), and _D<sub>2h</sub>_ (biaxial). The parameter file
[`params/D2h/D2h.ini`](params/D2h/D2h.ini) is intended to be used with the
`phase_diagram` classifier which maps the _(J<sub>1</sub>, J<sub>3</sub>)_
points to three labels, `O3`, `Dinfh`, and `D2h`, according to the known phase
diagram. The SVM will then solve three classification problems, yielding one
decision function to distinguish between each pair of the labels. Given that the
labels indeed correspond to the physical phases and the rank is chosen
appropriately (as is the case for the _D<sub>2h</sub>_ example), these three
decision functions can be related to the order parameter(s).

Note that the same samples can also be used with a different classifier.
Building the codes with the `GAUGE_CLASSIFIER` option set to `GRID`, the samples
will be labeled according to the grid points in phase diagram space they have
been sampled from. It is not necessary to rerun the sampling step; instead one
can resume from the previously written checkpoint with the executable compiled
with the `fixed_from_grid` classifer:

```bash
$ gauge-learn D2h.clone.h5 --skip-sampling --outputfile=D2h_grid.out.h5 --nu=0.1
```

The phase diagram has previously been sampled on a 10 x 10 grid, hence, the
samples are now assigned 100 distinct labels, resulting in 100 * 99 / 2 = 4950
decision functions. The result is written to the alternative location
`D2h_grid.out.h5` to avoid overwriting the previous result. It is obviously not
feasible to interpret each of the decision functions individually, but we can
use their bias parameters for the
[_graph analysis_](#spectral-graph-partitioning-analysis). To that end, a
moderate regularization level is appropriate.

**Command line options**

| Long flag                    | Short | Description                                                                                         |
|:-----------------------------|:-----:|:----------------------------------------------------------------------------------------------------|
| `--help`                     | `-h`  | Display ALPSCore help message (lists parameters)                                                    |
| `--skip-sampling`            | `-s`  | When restoring from checkpoint, skip the sampling stage and proceed to SVM optimization immediately |

Note that additionally [runtime parameters](#runtime-parameters) may also be
overridden using command line arguments.

### Testing: measuring decision function and observables

```bash
$ gauge-test Td.out.h5
```

will read the SVM model that was previously learned and measure its decision
function as an observable in independent Monte Carlo simulations at
`test.N_scan` equidistant points on the line connecting the points `test.a` and
`test.b` in the phase diagram to obtain the order parameter curve. (See
[below](#testing-stage) for the relevant parameters.) Results are summarized in
a text file `Td.test.txt` and full observables are stored in `Td.test.h5`.

The column layout of the file `Td.test.txt` is as follows:
* a number of columns, specifying the phase diagram point of the measurement,
  depending on the dimensionality of the phase diagram (one temperature for the
  Ising model, two couplings _J<sub>1</sub>_ and _J<sub>3</sub>_ for the gauge
  model),
* two columns holding the mean and error of the predicted (integer) SVM label,
  measured as an observable,
* for each decision function, two columns with the mean and error of the
  measured decision function value,
* for any other observable defined by the code's SVM interface functions (_i.e._
  magnetization, nematicity, etc.), two columns with the mean and error of the
  measured value.

The file `Td.test.txt` is suitable for plotting with `gnuplot`. A plot of the
decision function can be created using:

    gnuplot> plot 'Td.test.txt' using 2:5:6 with yerrorlines

![Decision function curve for the tetrahedral symmetry](doc/img/Td.png)

**Command line options**

| Long flag                    | Short | Description                                                       |
|:-----------------------------|:-----:|:------------------------------------------------------------------|
| `--help`                     | `-h`  | Display ALPSCore help message (lists parameters)                  |
| `--rescale`                  | `-r`  | Shift decision functions by the bias and rescale to unit interval |

Note that additionally [runtime parameters](#runtime-parameter) may also be
overridden using command line arguments.

### Extracting the coefficient matrix

```bash
$ gauge-coeffs Td.out.h5
```

reads the SVM model and contracts over the support vectors to extract the
coefficient matrix, block structure, and performs miscellaneous analyses and
processing steps, such as fitting and removing self-contractions or comparing to
the exact result.

The three panels in [Fig. 2 of the manuscript][1] can be reproduced by the
following invocations:

```bash
$ gauge-coeffs Td.out.h5 -u
$ gauge-coeffs Td.out.h5 -u --block=[lmn:lmn]
$ gauge-coeffs Td.out.h5 -u --block=[lmn:lmn] --remove-self-contractions
```

These options are detailed below.

**Index arrangement**

* `-u | --unsymmetrize`: coefficients involving redundant monomial
  (cf. [Supplementary Materials][1]) are reconstructed, i.e. the coefficient
  value of the corresponding non-redundant coefficient is copied.
* `-r | --raw`: the indicies are *not* rearranged. By default, the indices are
  reshuffled in the form _(α<sub>1</sub>, ..., α<sub>n</sub>, a<sub>1</sub>,
  ..., a<sub>n</sub>)_ and lexicographically ordered such that the color-block
  structure becomes apparent. Specifying this flag disables this, and indices
  are arranged as _(α<sub>1</sub>, a<sub>1</sub>, ..., α<sub>n</sub>,
  a<sub>n</sub>)_.

**Extracting individual blocks**

The program has two major modes: extraction of the full coefficient matrix and
*single-block mode* where a single block is targeted and extracted exclusively.
This is done by specifying the command line parameter `--block=<block-spec>`
where `<block-spec>` identifies a block by its color indices. E.g. in the case
of the tetrahedral (rank-3) order, the non-trival block can be found by
specifying `--block=[lmn:lmn]`.

**Contraction analysis**

* `-c | --contraction-weights`: for each block (or only the single block),
  perform a least-squares fit of contraction "masks" to obtain the coefficients
  with which each contraction contributes and output those.
* `-s | --remove-self-contractions`: perform the same analysis as above, but
  consequently isolate the contributions due to self-contractions and subtract
  them from the full coefficient matrix.

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

**Multiclassification**

When analyzing the result of a multiclassification problem (see the
_D<sub>2h</sub>_ example above), the analysis is performed for each decision
function separately. Because of the quadratic growth of the number of decision
functions, this can get messy for big multiclassification problems. Instead, one
can list the decision functions and their biases without extracting the
coefficient matrices using:

```bash
$ gauge-coeffs D2h.out.h5 --list
0:   O3 -- Dinfh       rho = -1.0052
1:   O3 -- D2h         rho = -1.00266
2:   Dinfh -- D2h      rho = -0.659734
```

In this lists, the decision functions are given a running number. This number
can be used to select a single decision function and extract only its
coefficient matrix. For example, to limit the analysis to the _O(3) /
D<sub>2h</sub>_ transition:

```bash
$ gauge-coeffs D2h.out.h5 -u --transition=1
```

**Command line options**

| Long flag                    | Short | Description                                                                                                                                                                   |
|:-----------------------------|:-----:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--help`                     | `-h`  | Display ALPSCore help message (lists parameters)                                                                                                                              |
| `--verbose`                  | `-v`  | Print information on currently running step                                                                                                                                   |
| `--list`                     | `-l`  | List the transitions between labels and their bias value, but to not write any files                                                                                          |
| `--transition=<num>`         | `-t`  | Rather than extracting coefficients for all transitions in a multiclassification setting, only consider the transition numbered `<num>` (as listed using the `--list` option) |
| `--unsymmetrize`             | `-u`  | Assign the extracted coefficient to _all_ equivalent monomials, rather than only to a single representative                                                                   |
| `--raw`                      | `-r`  | Indicies are *not* rearranged; redundant monomials are not included in the output pattern                                                                                     |
| `--blocks-only`              | `-b`  | Skip output of the full coefficient matrix and only output block structure; unavailable (and pointless) in single-block mode.                                                 |
| `--block=<block-spec>`       |       | Only output the coefficient matrix block identified by `<block-spec>`                                                                                                         |
| `--contraction-weights`      | `-c`  | For each block (or only the single block), perform a least-squares fit of contraction "masks" to obtain the contributions of each contraction and output those.               |
| `--remove-self-contractions` | `-s`  | Perform the same analysis as above, but consequently isolate the contributions due to self-contractions and subtract them from the full coefficient matrix.                   |
| `--exact`                    | `-e`  | Calculate and output the exact result for the coefficient matrix (if available)                                                                                               |
| `--diff`                     | `-d`  | Output the deviation from the exact result for the coefficient matrix and print total relative deviation _δ_                                                                  |
| `--result=<name>`            |       | The name of the exact result to look up; hardcoded values: `Cinfv`, `Dinfh`, `D2h`; automatically chosen according to `gauge_group` parameter                                 |

Short flags may be combined into one multi-flag, as is customary for
POSIX-compatible programs.

### Spectral graph partitioning analysis

The `gauge-segregate-phases` program is intended to be used in conjunction with
massive multiclassification programs (such as the previously constructed file
`D2h_grid.out.h5`) to infer the topology of the phase diagram based on the bias
criterion.

The rationale is, that if the bias of a decision function between two points is
far from one (the ideal value expected for a physical transitions), those two
points are either in the same phase, or the phase transition could not be
captured at the rank considered. We then go ahead and construct an undirected
simple graph by adding an edge connecting those two points (vertices). On the
other hand, if the bias is "close" to one, we assume there is a phase transition
taking place between the points and we don't put an edge. The threshold value
_ρ<sub>c</sub>_ can be tuned to control the inclusivity of the criterion: an
edge is included in the graph if the corresponding bias _ρ_ exceeds
_ρ<sub>c</sub>_: _ρ > ρ<sub>c</sub>_.

The resulting graph can be subjected to a spectral partitioning analysis. To
that end, the eigendecomposition of the [Laplacian][15] of the graph is
calculated. The degeneracy of the lowest eigenvalue (zero) indicates the number
of connected components within the graph. We recommend to choose _ρ<sub>c</sub>_
such that the graph consists of a single connected component, _i.e._ a slightly
larger value of _ρ<sub>c</sub>_ would lead to a degeneracy of eigenvalue zero.
For the above example of the _D<sub>2h</sub>_ symmetry, this sweet spot happens
to be achieved at around _ρ<sub>c</sub>=1.6_, but your mileage may vary:

```bash
$ gauge-segregate-phases D2h_grid.out.h5 --rhoc=1.7
Degeneracy of smallest eval: 2
$ gauge-segregate-phases D2h_grid.out.h5 --rhoc=1.6
Degeneracy of smallest eval: 1
```

Then, the second largest eigenvalue is the so-called _algebraic connectivity_
and its corresponding eigenvector, the _Fiedler vector_ can be used to infer the
phase diagram: if the graph features regions which are strongly intraconnected,
but weakly interconnected, the algebraic connectivity will be small and the
entries of the Fiedler vector will cluster such that the entries corresponding
to intraconnected regions will have a similar value.

Thus, the Fiedler vector can be visualized and will be reminiscient of the phase
diagram. To that end, the `*-segregate-phases` program outputs two files,
`edges.txt` and `phases.txt`. The former lists pairs of points corresponding to
the edges of the graph, the latter gives _all_ the eigenvectors of the Laplacian
matrix such that the second dataset (index 1) is the Fiedler vector. Using
gnuplot, we can see that we achieve a decent approximation of the
_D<sub>2h</sub>_ phase diagram:

    gnuplot> plot 'edges.txt' using 2:1 with lines, \
                  'phases.txt' index 1 using 2:1:3 with points pt 7 lc palette

![Graph and Fiedler vector](doc/img/graph.png)

**Command line options**

| Long flag                    | Short | Description                                            |
|:-----------------------------|:-----:|:-------------------------------------------------------|
| `--help`                     | `-h`  | Display ALPSCore help message (lists parameters)       |
| `--verbose`                  | `-v`  | Print information on currently running step            |
| `--rhoc=<number>`            | `-r`  | Set the cutoff bias of inclusion of edges in the graph |

Runtime parameters
------------------

This section lists all parameters that can be used to control the behavior of
the simulations with respect to runtime duration and output, model-specific
configuration, kernel mapping and regularization of the SVM optimization, the
way the parameter space is swept, and the way samples are labeled (classified)
before being fed to the SVM.

Some parameters feature a hierarchical name where levels in the hierarchy are
separated by dots. These parameters can also be group by hierarchical prefix.
The following three specification of the parameter `sweep.grid.N1` are
equivalent:

```ini
sweep.grid.N1 = 42

[sweep]
grid.N1 = 42

[sweep.grid]
N1 = 42
```

Note that the hierarchical prefix has to be specified in full and cannot be
nested. _I.e._, the following version **does not** work:

```ini
[sweep]
[grid]
N1 = 42                   # this is wrong!
```

### Simulation runtime

The following parameters control the runtime behavior of the `*-learn` and
`*-test` programs. In general, the program will run until either it finished the
specified amount of Monte Carlo sweeps, or until the wallclock `timelimit` is
exceed. In both cases, the program will write a checkpoint to disk which the
simulation can be resumed from (to complete the required number of sweeps, or to
perform additional sampling).

| Parameter name                                  | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `progress_interval`                             | `3`              | Time between progress reports during sampling (sec)    |
| `SEED`                                          | `42`             | Seed of the master thread PRNG                         |
| `timelimit`                                     | `0`              | Time limit before termination in sec (0 = indefinite)  |
| `outputfile`                                    | `*.out.h5`       | HDF5 output file name for SVM model                    |
| `checkpoint`                                    | `*.clone.h5`     | HDF5 checkpoint file name                              |
| `total_sweeps`                                  | `0`              | Number of MC steps per phase diagram point sampled     |
| `thermalization_sweeps`                         | `10000`          | Thermalization steps after each phase point change     |

### Phase diagram point specification

Depending on the physical model simulated by the client codes, the phase diagram
space (_i.e._ parameter space) may be different. For example, the `ising` code
uses only the temperature, so the parameter space is one-dimensional. The
`gauge` client code has a two-dimensional phase diagram spanned by the
_J<sub>1</sub>_ and _J<sub>3</sub>_ coupling constants.

Both the [sweep policies](#sweep-through-the-phase-diagram) and the
[classifiers](#classifiers) which define traversals and regions in the phase
diagram thus have parameters which depend on the (dimensionality of the) model's
phase diagram. In the relevant subsection, these will invoke the placeholder
`<phase-diag-point-spec>`. In case of the `gauge` client code, this placeholder
has to be replaced by the following two parameters:

| `J1J3` phase diagram point                      | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `J1`                                            | `0`              | _β J<sub>1</sub>_ coupling                             |
| `J3`                                            | `0`              | _β J<sub>3</sub>_ coupling                             |

Likewise, in the `ising` client code, `<phase-diag-point-spec>` has to be
replaced by a single parameter:

| `temperature` phase diagram point               | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `temp`                                          | `1`              | Temperature                                            |

Note that some sweep policies and classifiers are not generic for arbitrary
phase diagram points, but rather only work with the `temperature` type. These
are deprecated and will be removed in the next version. Their functionality can
be replicated by the generic versions. See the deprecation warnings below.

### SVM optimization (learning)

| Parameter name | Default    | Description                                                                                      |
|:---------------|:----------:|:-------------------------------------------------------------------------------------------------|
| `nu`           | `0.5`      | Regularization parameter _ν_ in [0, 1]                                                           |

Further, note that the parameters `rank`, `symmetrized`, `cluster`, and `color`
for the gauge client code affect the feature vectors fed to the SVM.


### Testing stage

During the testing stage (programs `*-test`) a line through the phase diagram
from points `a` to `b` is sampled at `N_scan` equidistant points and the SVM
decision function along with any analytical reference quantities are measured.

| Parameter name                                  | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `test.N_scan`                                   | `10`             | Number of (equidistant) points to measure              |
| `test.filename`                                 | `*.test.h5`      | HDF5 test file name                                    |
| `test.txtname`                                  | `*.test.txt`     | Human-readable ASCII test result file name             |
| `test.a.<phase-diag-point-spec>`                | _required_       | First end point of the line along which to test        |
| `test.b.<phase-diag-point-spec>`                | _required_       | Second end point of the line along which to test       |

Note that at each point, the simulation performs the regular amount of Monte
Carlo sweeps as determined by the `thermalization_sweeps` and `total_sweeps`
parameters. One may want to override these parameters from the command line if
the amount necessary for testing differs from that used in learning.

### Client code-specific parameters

This subsection lists the parameters which are specific to the client codes and
control the parameters of the Hamiltonian, as well as details of the Monte Carlo
update schemes.

**Ising client code**

| Parameter name                                  | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `length`                                        | _required_       | Linear system size                                     |
| `temperature`                                   | _required_       | Initial temperature (_not relevant for SVM use case_)  |

**Gauge client code**

| Parameter name            | Default          | Description                                                                                |
|:--------------------------|:----------------:|:-------------------------------------------------------------------------------------------|
| `length`                  | _required_       | Linear system size                                                                         |
| `gauge_group`             | _required_       | One of: `Cinfv`, `Dinfh`, `D2h`, `D2d`, `D3`, `D3h`, `T`, `Td`, `Th`, `O`, `Oh`, `I`, `Ih` |
| `group_size`              | _required_       | Number of elements in the gauge group (_has to be chosen accordingly_)                     |
| `O3`                      | `0`              | Whether the group is a subgroup of SO(3) (`0`) or just O(3) (`1`) (_choose accordingly_)   |
| `temperature`             | `1`              | Temperature (_rescales J<sub>1</sub>, J<sub>3</sub> couplings, keep at 1_)                 |
| `hits_R`                  | `1`              | Number of updates to local spins per MC unit step                                          |
| `hits_U`                  | `1`              | Number of updates to gauge fields per MC unit step                                         |
| `global_gauge_prob`       | `0.05`           | Probability to perform global gauge transformation                                         |
| `sweep_unit`              | `10`             | Number of unit steps per Monte Carlo step                                                  |
| `<phase-diag-point-spec>` | _optional_       | Initial phase diagram point (_not relevant for SVM use case_)                              |

In addition, the following parameters influence the monomial mapping of the spin
configuration, in particular its rank, redundancy, and the choice of the spin
cluster. Currently, these parameters are only applicable to the gauge code, but
will be elevated to transcend this specific client code in the future.

| Parameter name | Default    | Description                                                                                      |
|:---------------|:----------:|:-------------------------------------------------------------------------------------------------|
| `rank`         | _required_ | Rank of the monomial mapping                                                                     |
| `symmetrized`  | `1`        | Eliminate redundant (symmetric) monomials (`1`) or not (`0`)                                     |
| `color`        | `triad`    | Consider single spin per site (`mono`) or all three (`triad`)                                    |
| `cluster`      | `single`   | Use `single` spin cluster, `bipartite` lattice (two-spin cluster), or `full` spin configurations |

### Sweep through the phase diagram

At the sampling stage, the simulation performs a sweep through the phase diagram
space. The _sweep policy_ determines the strategy followed to generate these
points. Some are deterministic, such as `cycle` and `grid`, others are
nondeterministic and use pseudorandom numbers, like `uniform` and
`uniform_line`. The sweep policy can be selected at runtime through the
`sweep.policy` parameter.

Regardless of the sweep policy selected, it will be used to generate `sweep.N`
(possibly repeating) points. At each point, `sweep.samples` spin configurations
are sampled. These samples are evenly spaced during the `total_sweeps` number of
Monte Carlo steps at each point.

| Parameter name                                  | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `sweep.policy`                                  | `cycle`          | Name of sweep policy; see below                        |
| `sweep.N`                                       | `1000`           | Number of phase diagram points to be sampled           |
| `sweep.samples`                                 | `1000`           | Number of spin samples per phase diagram point         |

**Cycling through a set of predefined phase diagram points**

The `cycle` sweep policy can be used to probe the phase diagram at a set of up
to 8 points `P1`, ..., `P8` which are manually provided as individual
parameters. If `sweep.N` exceeds the number of points specified, it will cycle
back to the beginning (`P1`), hence the name. This can be useful when more than
eight threads ought to be used for sampling.

| `cycle` sweep policy                            | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `sweep.cycle.P<i>.<phase-diag-point-spec>`      | _optional_       | _`i`th_ phase diagram point in `cycle` sweep           |

**Equidistant rectangular grid**

The `grid` sweep policy imposes a regular grid on the region of the phase
diagram spanned by the two points `a` and `b`. The spacing of the grid points is
equidistant within each dimension of the phase diagram, where the spacing is
determined by the number of grid points in that dimension, `N1`, `N2`, ... .
Valid values for `N<d>` include `1`; `0` is not permitted.

Similar to the `cycle` policy, all grid points are traversed before cycling back
to the beginning.

| `grid` sweep policy                             | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `sweep.grid.N<d>`                               | `1`              | Number of grid points along the _`d`th_ dimension      |
| `sweep.grid.a.<phase-diag-point-spec>`          | _required_       | Lower-left corner of rectangular uniform grid          |
| `sweep.grid.b.<phase-diag-point-spec>`          | _required_       | Upper-right corner of rectangular uniform grid         |

**Uniform rectangular region**

The `uniform` sweep policy generates points which are uniformly, yet randomly,
distributed in the rectangular region spanned by the two points `a` and `b`.

| `uniform` sweep policy                          | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `sweep.uniform.a.<phase-diag-point-spec>`       | _required_       | Lower-left corner of rectangular region                |
| `sweep.uniform.b.<phase-diag-point-spec>`       | _required_       | Upper-right corner of rectangular region               |

**Uniform line segment**

The `uniform_line` sweep policy generate points uniformly on the line segment
connecting the points `a` and `b`. For one-dimensional phase diagrams, it is
identical to the `uniform` sweep policy.

| `uniform_line` sweep policy                     | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `sweep.uniform_line.a.<phase-diag-point-spec>`  | _required_       | First end point of the uniformly sampled line          |
| `sweep.uniform_line.b.<phase-diag-point-spec>`  | _required_       | Second end point of the uniformly sampled line         |

#### Sweeps through 1-dim. `temperature` phase diagram

The following sweep policies are only available in conjunction with the
`temperature` phase space point (_i.e._ for the Ising model) and are not
generic. They were designed to change the temperature successively by small
amounts to limit the need for intermediate thermalization. This turned out to be
a non-issue. These specialized sweep policies are deprecated at this point and
will be removed in an upcoming version.

**Gaussian temperature distribution**

The `gaussian_temperatures` sweep policy perform a random walk within the
interval [`temp_min`,`temp_max`] where at each step, the temperature is changed
by at most `temp_step`. When converged, the resulting samples will be
approximately distributed according to a Gaussian distribution around
`temp_center` with standard deviation `temp_sigma`.

The rationale was to have more points close to the critical temperature as those
might be the most useful to define the order. This did not turn out to be the
case.

| `gaussian_temperatures` sweep policy            | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `sweep.gaussian_temperatures.temp_step`         | `1`              | Maximum change of the temperature per Markow step      |
| `sweep.gaussian_temperatures.temp_center`       | `1`              | Mean of the gaussian temperature distribution          |
| `sweep.gaussian_temperatures.temp_sigma`        | `1`              | Standard deviation  of gaussian distribution           |
| `sweep.gaussian_temperatures.temp_min`          | `0`              | Minimum temperature permissible in Markow chain        |
| `sweep.gaussian_temperatures.temp_max`          | ∞                | Maximum temperature permissible in Markow chain        |

**Uniform temperature distribution**

The `uniform_temperatures` sweep policy performs a random walk in the interval
`[temp_min`, `temp_max`] where at each step, the temperature is changed by at
most `temp_step`. The temperatures are otherwise approximately uniformly
distributed.

As thermalization is not a critical issue for reasonable temperatures, the
generic `uniform` (or `uniform_line`) sweep policy can be used to achieve the
same result.

| `uniform_temperatures` sweep policy             | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `sweep.uniform_temperatures.temp_step`          | `1`              | Maximum change of the temperature per Markdow step     |
| `sweep.uniform_temperatures.temp_min`           | `0`              | Minimum temperature permissible in Markow chain        |
| `sweep.uniform_temperatures.temp_max`           | ∞                | Maximum temperature permissible in Markow chain        |

**Equidistant temperature distribution**

The `equidistant_temperatures` sweep policy generates equidistant samples within
the interval [`temp_min`, `temp_max`].

The generic `grid` policy can be used to achieve the same behavior.

| `equidistant_temperatures` sweep policy         | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `sweep.equidistant_temperatures.temp_min`       | `0`              | Minimum temperature permissible in Markow chain        |
| `sweep.equidistant_temperatures.temp_max`       | ∞                | Maximum temperature permissible in Markow chain        |

### Classifiers

Simulation samples need to be labeled in some fashion and these labels are
exposed to the SVM. Thus, some sort of _classification_ or mapping of phase
diagram space points to a finite number of labels needs to be specified.
Depending on the model and the available physical information, different choices
for this _classifier_ might be appropriate. A number of those are documented in
this section and users may define additional ones without too much effort.

Since the label type that the classifier outputs needs to be known at compile
time (as fixed size labels are treated differently by the SVM wrapper library),
the classifier cannot be switched at runtime (unlike the sweep policy). In
client codes supporting this, one may switch between various classifiers by
setting the appropriate value of the `*_CLASSIFIER` build option (_e.g._
`GAUGE_CLASSIFIER`).

However, since this classification step happens only after sampling and
immediately prior to the SVM optimization, one can recompile the `*-learn`
program with a different classifer and repeat the SVM optimization on the same
data but with different labeling by restoring the `*.clone.h5` file and skipping
further sampling, _e.g._:

```bash
$ gauge-learn Td.clone.h5 --skip-sampling
```

_(Note: in version 1 of this code, only the temperature was used as a parameter
of the gauge model. The notion of different classifiers was absent and a
critical temperature was used exclusively for binary classification; see below
at the end.)_

**Classification based on known phase diagram**

When the phase diagram is already known, the user will likely want to label
points in the phase diagram space accordingly. This can be achieved with the
`phase_diagram` classifier. The phase diagram is encoded as a collection of
disjoint polygons which correspond to the phases and cover the relevant parts of
the phase diagram space; each point is given the label associated with the
polygon it is contained in.

The available phase diagrams are hard-coded but additional ones can be added
relatively easily. The desired diagram is selected from the "database" by
specifying its name:

| `phase_diagram` classifier parameters           | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `classifier.phase_diagram.name`                 | _required_       | Name of the known phase diagram to retrieve            |

Each phase diagram has an associated label type that enumerates the phases.
Thus, phase diagrams of different systems have typically different phases and it
is required that those are selected appropriately _at compile time_ through
different choices of the `*_CLASSIFIER` variable. Additionally, one may want to
compare the performance of different phase diagram candidates for the the
system. In the latter case, they correspond to the same `phase_diagram`
instantiation and can be switched between using the `name` parameter at runtime.

Phase diagrams are currently only used for the gauge client code. Below is a
table of supported combinations of build and runtime options, along with the
associated label types:

| Build option `*_CLASSIFIER` | Runtime parameter `name` | `label::*` | Label values                        |
|:----------------------------|:-------------------------|:-----------|:------------------------------------|
| `D2h`                       | `"D2h"`                  | `D2h`      | `D2h::O3`, `D2h::Dinfh`, `D2h::D2h` |
| `D2h`                       | `"D2h_Ke"`               | `D2h`      | _see above_                         |
| `D3h`                       | `"D3h"`                  | `D3h`      | `D3h::O3`, `D3h::Dinfh`, `D3h::D3h` |

**Classification according to sweep policy**

When the phase diagram is not yet known, the easiest approach is to sweep the
phase diagram with the `cycle` or `grid` policies to examine interesting points
or cover the space uniformly and then classify the samples with a running label
according to which point they are sampled from. Thus, no two different points in
the phase diagram will be given the same label, ruling out wrong assumptions on
the tentative phase diagram. One arrives at a (potentially big)
multiclassification problem and can use the result to decide which phase diagram
points should in fact be given the same label, either through manual inspection
(cf. Sec. VI.A of [arXiv:1810.05538][10]) or on a larger scale using the graph
analysis (cf. Sec. VII of [arXiv:1810.05538][10]).

The corresponding classifiers are called `fixed_from_cycle` and
`fixed_from_grid` and are enabled by setting the `*_CLASSIFIER` build option to
`CYCLE` or `GRID`, respectively.

**Binary classification using a hyperplane through the phase diagram**

When samples from multiple (possibly random) points in the phase diagram are
used to analyze a particular transition, the `hyperplane` classifier allows for
a binary classification by partitioning the phase diagram space into two regions
on either side of a "hyperplane" (a single point in 1d phase diagrams, a line in
2d, a plane in 3d, ...). The hyperplane is specified by any one point on the
plane (fixing its offset from the origin) and a normal vector which is
orthogonal to the plane and does not need to be normalized.

The resulting label `label::binary` has the two values `ORDERED` and
`DISORDERED`. The normal vector is pointing into the half space that is
`ORDERED`. Hyperplane classification is enabled by selecting `HYPERPLANE` as the
`*_CLASSIFIER` build option.

| `hyperplane` classifier parameters                      | Default    | Description                                 |
|:--------------------------------------------------------|:----------:|:--------------------------------------------|
| `classifier.hyperplane.support.<phase-diag-point-spec>` | _required_ | A point somewhere on the desired hyperplane |
| `classifier.hyperplane.normal.<phase-diag-point-spec>`  | _required_ | The normal vector defining the orientation  |

**Classifiers for `temperature` phase diagram points**

One special instance of binary classification is the in one-dimensional
temperature phase diagram space of the Ising model. The Ising client code uses
the `critical_temperature` classifier which does pretty much the same as a
hyperplane classifier but is more simple and less versatile. For the Ising code,
this choice is hardcoded.

| `critical_temperature` classifier parameters    | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `classifier.critical_temperature.temp_crit`     | `1`              | Critical temperature                                   |


Client code API
---------------

Client codes have to conform to ALPSCore's model for Monte Carlo simulations. On
top of that, we require that the simulation class (which is derived from
`alps::mcbase`) publicly exposes four member types:

>>>
```cpp
using phase_classifier = /* ... */;
using phase_point = phase_classifier::point_type;
using phase_label = phase_classifier::label_type;
using phase_sweep_policy_type = phase_space::sweep::policy<phase_point>;
```
>>>

As indicated, the types `phase_point` and `phase_label` need to be consistent
with the corresponding member types of `phase_classifier`. Note that while the
`phase_classifier` type may be changed (using build options like
`GAUGE_CLASSIFIER` in the gauge code) and the `phase_label` would change
accordingly, the `phase_point` type would typically always be the same for a
given simulation. Then, the `*-learn` program can be restored from a checkpoint
file (`*.clone.h5`) written by a different version of the program that is using
a different classifier.

The last type, `phase_sweep_policy_type` is the common abstract base class for
all sweep policies in the phase diagram space of the model. It is not strictly
required, but included here since it appears in the signatures below.

In addition to these types, the simulation class must implement the following
member functions:

>>>
```cpp
Container const& configuration() const;
```

returns the feature vector that is passed on to the quadratic SVM kernel. Note
that the monomial mapping for high-rank order parameters has to have been
carried out by the client code in this version. This is going to change upcoming
versions. Currently, the gauge code contains this logic.
>>>

>>>
```cpp
size_t configuration_size() const;
```

returns the size of the feature vector. Must be identical to
`configuration().size()`.
>>>

>>>
```cpp
phase_point phase_space_point() const;
```

returns the current location in the phase diagram space.
>>>

>>>
```cpp
void update_phase_point(phase_sweep_policy_type & sweep_policy);
```

updates (or attempts to do so) the current location in the phase diagram space.
>>>

>>>
```cpp
void reset_sweeps(bool skip_therm = false);
```

resets the internal counter of the simulation progress; called after phase point
update. The optional argument indicates if the phase point in fact has not
changed and thus no intermediate thermalization phase is necessary.
>>>

>>>
```cpp
bool is_thermalized() const;
```

returns whether the simulation is thermalized.
>>>

>>>
```cpp
std::vector<std::string> order_param_names() const;
```

returns a vector of names of registered observables (which have been added to
the inherited `measurements` object) which are to be considered as reference
observables to compare to. These are then measured in the `*-test` program.
>>>

Note that this interface is subject to (breaking) change in upcoming versions.

Changelog
---------

### Changes from version 1

* Support for multiclassification
  - in upstream SVM wrapper library
  - all decision functions are being measured in `*-test`
* Gauge client code uses both _J<sub>1</sub>_ and _J<sub>2</sub>_ as parameters
  which locate the model in the phase diagram
  - introduced _sweep policy_ concept to traverse phase diagram in varied ways
  - introduced _classifier_ concept to map points in the phase diagram to labels
    which are used as input for the SVM
* Spectral graph partitioning analysis using separate program:
  `*-segregate-phases`

License
-------

Copyright © 2018-2019  Jonas Greitemann, Ke Liu, and Lode Pollet

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
[7]: https://github.com/ALPSCore/ALPSCore/blob/master/INSTALL.md
[8]: https://eigen.tuxfamily.org/
[9]: https://doi.org/10.1103/PhysRevB.99.060404
[10]: https://arxiv.org/abs/1810.05538
[11]: https://doi.org/10.1103/PhysRevB.99.104410
[12]: https://github.com/adishavit/argh
[13]: https://doi.org/10.1103/PhysRevB.96.205146
[14]: https://doi.org/10.1103/PhysRevX.6.041025
[15]: https://en.wikipedia.org/wiki/Laplacian_matrix
