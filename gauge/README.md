Lattice Gauge Model Client Code
===============================

tba


Building and Installation
-------------------------

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

Runtime parameters
------------------

### Simulation runtime

| Parameter name                                  | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `total_sweeps`                                  | `0`              | Number of MC steps per phase diagram point sampled     |
| `thermalization_sweeps`                         | `10000`          | Thermalization steps after each phase point change     |

### Model parameters

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
| `<phase-diag-point-spec>` | _optional_       | Initial phase diagram point (_not relevant for TKSVM use case_)                            |

### Tensorial kernel

| Parameter name | Default    | Description                                                                                      |
|:---------------|:----------:|:-------------------------------------------------------------------------------------------------|
| `rank`         | _required_ | Rank of the monomial mapping                                                                     |
| `symmetrized`  | `1`        | Eliminate redundant (symmetric) monomials (`1`) or not (`0`)                                     |
| `color`        | `triad`    | Consider single spin per site (`mono`) or all three (`triad`)                                    |
| `cluster`      | `single`   | Use `single` spin cluster, `bipartite` lattice (two-spin cluster), or `full` spin configurations |

### Phase diagram point specification

<!-- This is `<phase-diag-point-spec>`. -->

| `J1J3` phase diagram point                      | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `J1`                                            | `0`              | _β J<sub>1</sub>_ coupling                             |
| `J3`                                            | `0`              | _β J<sub>3</sub>_ coupling                             |
