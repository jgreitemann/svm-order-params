Ising Model Client Code
=======================

This client code is adapted from the [two-dimensional Ising model tutorial][1]
provided by ALPSCore to demonstrate how a basic ALPSCore-based simulation may be
used together with the TK-SVM framework.

Famously, the 2d Ising model exhibits a phase transition at _T<sub>c</sub> =
2.269 J_ to a ferromagnetic phase. Ponte and Melko demonstrated that an SVM with
a quadratic kernel may be used to learn this simple magnetic order [[Phys. Rev.
B **96**, 205146 (2017)][2]]. Step-by-step instructions on how to reproduce
their results are included as comments in the example parameter file
[`params/ising.ini`](params/ising.ini).

Building and Installation
-------------------------

CMake is used to build this code:
```bash
$ cd svm-order-params/ising
$ mkdir build && cd build
$ cmake ..
$ make -jN
```

Finally, using `make install`, the compiled executables can be copied to the
`bin` directory at the location configured in `CMAKE_INSTALL_PREFIX`. This step
is optional.

Refer to the [top-level README](../README.md) for information on dependencies
and cloning of submodules.

Runtime parameters
------------------

This sections lists the runtime parameters which are defined by — and exclusive
to — this client code. These parameters supplement those lists in the section
[→ Runtime parameters](../README.md#runtime-parameters) of the top-level README.

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

Note that the [`tksvm::phase_space::point::temperature`
type](../include/tksvm/phase_space/point/temperature.hpp) is used to encapsulate
the one-dimensional (temperature) parameter space of the phase diagram. Its
parameter specification is given in the [corresponding
section](../README.md#phase-diagram-point-specification) of the top-level README
file.

### Tensorial kernel

| Parameter name | Default    | Description                                                                                      |
|:---------------|:----------:|:-------------------------------------------------------------------------------------------------|
| `rank`         | _required_ | Rank of the monomial mapping                                                                     |
| `symmetrized`  | `1`        | Eliminate redundant (symmetric) monomials (`1`) or not (`0`)                                     |

[1]: https://github.com/ALPSCore/ALPSCore/tree/master/tutorials/mc/ising2_mc
[2]: https://doi.org/10.1103/PhysRevB.96.205146
