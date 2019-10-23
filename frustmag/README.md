Generic Client Code for Frustrated Spin Models
==============================================

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
| `total_sweeps`                                  | `0`              | Number of MC steps per phase diagram point sampled     |
| `lattice.bravais.length`                        | _required_       | Linear size of the bravais lattice in units of unitcells |
| `lattice.bravais.periodic`                      | `true`           | Boundary conditions on the simulation volume (PBC = `true`, OBC = `false`) |

### Update scheme

| Parameter name                                  | Default          | Description                                            |
|:------------------------------------------------|:----------------:|:-------------------------------------------------------|
| `total_sweeps`                                  | `0`              | Number of MC steps per phase diagram point sampled     |
| `update.single_flip.cos_theta_0`                | `-1`             | cos(θ<sub>0</sub>) |
| `pt.update_sweeps`                              | ∞                | Number of MC updates between PT updates                |
| `pt.query_freq`                                 | `1`              | Number of PT queries in one PT update cycle            |

### Tensorial kernel

| Parameter name | Default    | Description                                                                                      |
|:---------------|:----------:|:-------------------------------------------------------------------------------------------------|
| `rank`         | _required_ | Rank of the monomial mapping                                                                     |
| `symmetrized`  | `1`        | Eliminate redundant (symmetric) monomials (`1`) or not (`0`)                                     |
| `cluster`      | `lattice`  | `single` or `lattice`                              |

