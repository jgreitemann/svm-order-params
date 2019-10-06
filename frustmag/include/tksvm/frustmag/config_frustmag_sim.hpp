// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2018-2019  Jonas Greitemann, Ke Liu, and Lode Pollet

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <tksvm/frustmag/frustmag_sim.hpp>
#include <tksvm/frustmag/hamiltonian/heisenberg.hpp>
#include <tksvm/frustmag/hamiltonian/ising.hpp>
#include <tksvm/frustmag/lattice/chain.hpp>
#include <tksvm/frustmag/lattice/dice.hpp>
#include <tksvm/frustmag/lattice/honeycomb.hpp>
#include <tksvm/frustmag/lattice/kagome.hpp>
#include <tksvm/frustmag/lattice/ortho.hpp>
#include <tksvm/frustmag/lattice/triangular.hpp>
#include <tksvm/frustmag/update/heatbath.hpp>
#include <tksvm/frustmag/update/mux.hpp>
#include <tksvm/frustmag/update/overrelaxation.hpp>
#include <tksvm/frustmag/update/single_flip.hpp>


namespace tksvm {
namespace frustmag {

#if defined HEISENBERG_HAMILTONIAN
template <template <typename> typename Lattice>
using hamiltonian_t_t = hamiltonian::heisenberg<Lattice>;
#elif defined ISING_HAMILTONIAN
template <template <typename> typename Lattice>
using hamiltonian_t_t = hamiltonian::ising<Lattice>;
#else
#error Unknown hamiltonian
#endif

#if defined CHAIN
using hamiltonian_t = hamiltonian_t_t<lattice::chain>;
#elif defined SQUARE
using hamiltonian_t = hamiltonian_t_t<lattice::square>;
#elif defined CUBIC
using hamiltonian_t = hamiltonian_t_t<lattice::cubic>;
#elif defined TRIANGULAR
using hamiltonian_t = hamiltonian_t_t<lattice::triangular>;
#elif defined HONEYCOMB
using hamiltonian_t = hamiltonian_t_t<lattice::honeycomb>;
#elif defined KAGOME
using hamiltonian_t = hamiltonian_t_t<lattice::kagome>;
#elif defined DICE
using hamiltonian_t = hamiltonian_t_t<lattice::dice>;
#else
#error Unknown lattice
#endif

static_assert(!is_serializable<typename hamiltonian_t::lattice_type>::value,
              "lattice is serializable, but shouldn't be");
static_assert(is_archivable<typename hamiltonian_t::lattice_type>::value,
              "lattice is not archivable");

#ifdef USE_CONCEPTS
namespace {
    template <typename T>
    requires Lattice<T>
    struct check_lattice {};
    template struct check_lattice<typename hamiltonian_t::lattice_type>;

    template <typename T>
    requires Hamiltonian<T>
    struct check_hamiltonian {};
    template struct check_hamiltonian<hamiltonian_t>;

    template <typename U>
    requires MetropolisUpdate<U>
    struct check_update {};
    template struct check_update<update::single_flip<hamiltonian_t>>;
}
#endif

template <typename Hamiltonian>
struct update_t {
    template <typename LatticeH>
    using type = update::muxer<
        update::single_flip
        , update::global_trafo
        , update::parallel_tempering
        >::type<LatticeH>;
};

template <template <typename> typename Lat>
struct update_t<hamiltonian::heisenberg<Lat>> {
    template <typename LatticeH>
    using type = update::muxer<
        update::heatbath
        , update::overrelaxation
        , update::global_trafo
        , update::parallel_tempering
        >::type<LatticeH>;
};

}

using sim_base = frustmag::frustmag_sim<frustmag::hamiltonian_t,
                                        frustmag::update_t<frustmag::hamiltonian_t>::type>;

}
