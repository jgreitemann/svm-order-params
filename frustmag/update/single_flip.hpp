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

#include <iterator>
#include <random>

namespace update {

template <typename Lattice>
struct single_flip_proposal {
    using site_iterator = typename Lattice::iterator;
    using site_state_type = typename Lattice::value_type;

    site_iterator site_it;
    site_state_type flipped;
};

template <typename LatticeHamiltonian>
struct single_flip {
private:
    using lattice_type = typename LatticeHamiltonian::lattice_type;
    using site_iterator = typename lattice_type::iterator;
    using site_state_type = typename lattice_type::value_type;
public:
    template <typename RNG>
    void update(LatticeHamiltonian & hamiltonian, RNG & rng) const {
        size_t lsize = hamiltonian.lattice().size();
        for (size_t j = 0; j < lsize; ++j) {
            size_t i = std::uniform_int_distribution<size_t>{0, lsize - 1}(rng);
            site_iterator site_it = std::next(hamiltonian.lattice().begin(), i);
            hamiltonian.metropolis({site_it, site_it->flipped(rng)}, rng);
        }
    }
};

}
