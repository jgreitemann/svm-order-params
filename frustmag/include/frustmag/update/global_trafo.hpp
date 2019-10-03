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

struct global_trafo_proposal {};

template <typename LatticeH>
struct global_trafo {
#ifdef USE_CONCEPTS
    static_assert(LatticeHamiltonian<LatticeH>,
                  "LatticeH is not a LatticeHamiltonian");
#endif
public:
    using hamiltonian_type = LatticeH;
    using acceptance_type = std::array<double, 1>;

    static void define_parameters(alps::params & parameters) {}

    template <typename... Args>
    global_trafo(alps::params const&, Args &&...) {}

    template <typename RNG>
    // requires SiteState<site_state_type, RNG>
    acceptance_type update(LatticeH & hamiltonian, RNG & rng) {
        hamiltonian.metropolis(global_trafo_proposal{}, rng);
        return {1};
    }
};

}
