Ising Model Client Code
=======================

tba

Using `temperature` phase diagram point (LINK).


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

| Parameter name                                  | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `length`                                        | _required_       | Linear system size                                     |
| `temperature`                                   | _required_       | Initial temperature (_not relevant for SVM use case_)  |

### Tensorial kernel

| Parameter name | Default    | Description                                                                                      |
|:---------------|:----------:|:-------------------------------------------------------------------------------------------------|
| `rank`         | _required_ | Rank of the monomial mapping                                                                     |
| `symmetrized`  | `1`        | Eliminate redundant (symmetric) monomials (`1`) or not (`0`)                                     |
