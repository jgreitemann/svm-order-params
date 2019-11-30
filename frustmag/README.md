Generic Client Code for Frustrated Spin Models
==============================================

The `frustmag` client code is a generic Monte Carlo code for classical spin
systems on frustrated lattices. It is highly modular and aims to allow for
composability between different Hamiltonians (interactions) with different
lattice geometries. This is enabled by a [generic Bravais lattice
class](include/tksvm/frustmag/lattice/bravais.hpp) which allows for the
iteration over either individual spins or lattice unit cells, and over nearest
neighbors of each spin. Periodic and open boundary conditions are both
supported. This way, both the interactions and most of the [→ update
schemes](#update-schemes) can be implemented in a lattice-agnostic fashion. Type
traits and template specialization are used to enable updates at compile time
when applicable.

This code was used to study the classical Heisenberg model on the Kagome lattice
using TK-SVM, the results of which are presented in chapter 9 of [J.
Greitemann's PhD thesis][1].

Building and Installation
-------------------------

CMake is used to build this code:
```bash
$ cd svm-order-params/frustmag
$ mkdir build && cd build
$ cmake ..
$ make -jN
```

Finally, using `make install`, the compiled executables can be copied to the
`bin` directory at the location configured in `CMAKE_INSTALL_PREFIX`. This step
is optional.

Refer to the [top-level README](../README.md) for information on dependencies
and cloning of submodules.

#### Build configuration

In addition to the [build configuration options of the TK-SVM
framework](../README.md#build-configuration), this code allows for the selection
of the model Hamiltonian and lattice geometry through the following build
options:

| Variable name          | Possible values                                                                     |
|:-----------------------|:------------------------------------------------------------------------------------|
| `HAMILTONIAN`          | `heisenberg` (_default_), `ising`                                                   |
| `LATTICE`              | `chain`, `square`, `cubic`, `triangular`, `honeycomb`, `kagome` (_default_), `dice` |

To customize a variable, pass the appropriate flag on to CMake, _e.g._ the
default configuration is explicitly stated as:
```bash
$ cmake -DHAMILTONIAN=heisenberg -DLATTICE=kagome ..
```
or use the interactive `ccmake` configurator.


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
| `total_sweeps`                                  | `0`              | Number of MC steps per phase diagram point sampled     |
| `lattice.bravais.length`                        | _required_       | Linear size of the bravais lattice in units of unitcells |
| `lattice.bravais.periodic`                      | `true`           | Boundary conditions on the simulation volume (PBC = `true`, OBC = `false`) |

### Update schemes

A number of different updates are implemented. Some of these are only possible
in some Hamiltonians or lattice geometries. See [J. Greitemann's PhD thesis][1],
chapter 3, for detailed descriptions of each of the following updates.

**Metropolis single spin flip (`single_flip`)**

Randomly picks a number of single spins, corresponding to the total number of
spins in the system, and attempts to update them with Metropolis probability.
For Ising-type (_Z<sub>2</sub>_) spins, the spin is attempted to be flipped. For
"_O(3)_" spins which are used in the Heisenberg model, by default a random spin
is drawn uniformly from the unit sphere. In the latter case _only_, an
additional parameter allows to instead draw the spin uniformly from a cone
centered around the previous value of the spin, such that the original and
tentatively updated spin span an angle of at most _θ<sub>0</sub>_.

| Parameter name                   | Default | Description          |
|:---------------------------------|:-------:|:---------------------|
| `update.single_flip.cos_theta_0` | `-1`    | cos(_θ<sub>0</sub>_) |

**Heat-bath algorithm (`heatbath`)**

Randomly picks a number of single spins, corresponding to the total number of
spins in the system, and updates them according to heat-bath probability. This
update is currently only implemented for the Heisenberg model which constitutes
a special case where the heat-bath algorithm can be made rejection-free.
This update does not have any parameters associated with it.

**Global transformation (`global_trafo`)**

Applies the same transformation to all spins. For the Ising model, all spins are
flipped; for the Heisenberg model, all spins are rotated by a random _O(3)_
matrix. As both are symmetries of their respective Hamiltonians, this update is
microcanonical and can thus always be accepted. Note that it does not help to
reduce autocorrelation times of physical observables which preserve the same
symmetry. It can however be beneficial in conjunction with TK-SVM, as these
symmetries are not explicitly "taught" to the machine. This update does not have
any parameters associated with it.

**Overrelaxation update (`overrelaxation`)**

The overrelaxation update is, too, a microcanonical update and can thus always
be accepted. It can only be formulated for continuous spin degrees of freedom
and is thus currently only available in the Heisenberg model. It randomly picks
a number of single spins, corresponding to the total number of spins in the
system, and updates them in a direction orthogonal to the local magnetization at
that site such that the total energy remains unchanged. This can again help
navigate the highly degenerate energy landscape of frustrated magnets.
This update does not have any parameters associated with it.

**Parallel tempering (`parallel_tempering`)**

This code also implements a parallel tempering update using the facilities
provided by the `pt_adapter` class. Refer to the [section on Parallel
Tempering](../README.md#parallel-tempering) in the top-level README file. The
[`parallel_tempering`](include/tksvm/frustmag/update/parallel_tempering.hpp)
update in this code merely initiates the PT update by calling
`negotiate_update`. The callback to calculate the logarithmic weight is deferred
to the Hamiltonian class. Both the Ising and Heisenberg model implement it.

The following parameters are used to control the frequency of the PT update.
Only every `pt.update_sweeps` updates is a PT update actually initiated. By
default, this _never_ happens! As PT is a rather expensive proposition, the user
must enable it explicitly by setting this parameter to a finite value. The
second parameter, `pt.query_freq` determines how often within the duration of
`pt.update_sweeps` each process should _check_ whether it has been requested for
an update. In computer systems with a fast interconnect, checking multiple times
between PT updates may help reduce waiting times.

| Parameter name     | Default | Description                                 |
|:-------------------|:-------:|:--------------------------------------------|
| `pt.update_sweeps` | ∞       | Number of MC updates between PT updates     |
| `pt.query_freq`    | `1`     | Number of PT queries in one PT update cycle |

### Tensorial kernel

The two options of the `cluster` parameter determine the choice of the spin
cluster for the evaluation of the tensorial kernel. `single` corresponds to a
single spin cluster while `lattice` uses the unitcells of the Bravais lattice as
spin clusters.

| Parameter name | Default    | Description                                                  |
|:---------------|:----------:|:-------------------------------------------------------------|
| `rank`         | _required_ | Rank of the monomial mapping                                 |
| `symmetrized`  | `1`        | Eliminate redundant (symmetric) monomials (`1`) or not (`0`) |
| `cluster`      | `lattice`  | `single` or `lattice`                                        |


[1]: https://nbn-resolving.org/urn:nbn:de:bvb:19-250579
