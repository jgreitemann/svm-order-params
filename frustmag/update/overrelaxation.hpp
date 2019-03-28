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

#include "concepts.hpp"

#include <alps/params.hpp>

namespace update {

template <typename Lat>
struct overrelaxation_proposal {
#ifdef USE_CONCEPTS
    static_assert(Lattice<Lat>, "Lat is not a Lattice");
#endif
    using site_iterator = typename Lat::iterator;

    site_iterator site_it;
};

template <typename LatticeH>
struct overrelaxation {
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
    using acceptance_type = std::array<double, 1>;

    static void define_parameters(alps::params & parameters) {}

    template <typename... Args>
    overrelaxation(alps::params const&, Args &&...) {}

    template <typename RNG>
    // requires SiteState<site_state_type, RNG>
    acceptance_type update(LatticeH & hamiltonian, RNG & rng) {
        auto site_it = hamiltonian.lattice().begin();
        for (; site_it != hamiltonian.lattice().end(); ++site_it) {
            hamiltonian.metropolis(overrelaxation_proposal<lattice_type>{site_it},
                                   rng);
        }
        return {1};
    }
};

}
