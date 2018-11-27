// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2018  Jonas Greitemann, Ke Liu, and Lode Pollet

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

#include "concepts.hpp"
#include "site/spin_O3.hpp"

#include <alps/params.hpp>

#include <array>
#include <iterator>
#include <random>
#include <type_traits>

namespace update {

template <typename Lat>
struct single_flip_proposal {
#ifdef USE_CONCEPTS
    static_assert(Lattice<Lat>, "Lat is not a Lattice");
#endif
    using site_iterator = typename Lat::iterator;
    using site_state_type = typename Lat::value_type;

    site_iterator site_it;
    site_state_type flipped;
};

template <typename LatticeH>
struct single_flip {
#ifdef USE_CONCEPTS
    static_assert(LatticeHamiltonian<LatticeH>,
                  "LatticeH is not a LatticeHamiltonian");
#endif
private:
    using lattice_type = typename LatticeH::lattice_type;
    using site_iterator = typename lattice_type::iterator;
    using site_state_type = typename lattice_type::value_type;
public:
    using hamiltonian_type = LatticeH;
    using proposal_type = single_flip_proposal<lattice_type>;
    using acceptance_type = std::array<double, 1>;

    static void define_parameters(alps::params & parameters) {
        if (std::is_same<site_state_type, site::spin_O3>::value)
            parameters.define<double>("update.single_flip.cos_theta_0", -1.,
                                      "cos(theta_0) \\in [-1; 1]");
    }

    single_flip(alps::params const& parameters)
        : cos_theta_0 {
            std::is_same<site_state_type, site::spin_O3>::value ?
                parameters["update.single_flip.cos_theta_0"] : -1.
        }
    {
    }

    template <typename RNG, typename..., typename S = site_state_type,
              typename = std::enable_if_t<std::is_same<S, site::spin_O3>::value>>
    // requires SiteState<site_state_type, RNG>
    acceptance_type update(LatticeH & hamiltonian, RNG & rng) {
        size_t lsize = hamiltonian.lattice().size();
        double acc = 0;
        for (size_t j = 0; j < lsize; ++j) {
            size_t i = std::uniform_int_distribution<size_t>{0, lsize - 1}(rng);
            site_iterator site_it = std::next(hamiltonian.lattice().begin(), i);
            if (hamiltonian.metropolis(
                    {site_it, site_it->flipped(rng, cos_theta_0)}, rng))
                acc += 1.;
        }
        return {acc / lsize};
    }

    template <typename RNG, typename..., typename S = site_state_type,
              typename = std::enable_if_t<!std::is_same<S, site::spin_O3>::value>,
              int dummy = -1>
    // requires SiteState<site_state_type, RNG>
    acceptance_type update(LatticeH & hamiltonian, RNG & rng) {
        size_t lsize = hamiltonian.lattice().size();
        double acc = 0;
        for (size_t j = 0; j < lsize; ++j) {
            size_t i = std::uniform_int_distribution<size_t>{0, lsize - 1}(rng);
            site_iterator site_it = std::next(hamiltonian.lattice().begin(), i);
            if (hamiltonian.metropolis(
                    {site_it, site_it->flipped(rng)}, rng))
                acc += 1.;
        }
        return {acc / lsize};
    }

private:
    double cos_theta_0 = -1;
};

}
