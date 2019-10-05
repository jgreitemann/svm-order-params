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

#include <algorithm>

#include <combinatorics/binomial.hpp>
#include <combinatorics/number_of_permutations.hpp>

#include <tksvm/utilities/indices.hpp>


namespace tksvm {
namespace symmetry_policy {

    struct symmetrized {
        size_t size (size_t range, size_t rank) const {
            return combinatorics::binomial(rank + range - 1, rank);
        }

        void advance_ind (indices_t & ind, size_t range) const {
            auto rit = ind.rbegin();
            ++(*rit);
            while (*rit == range) {
                ++rit;
                if (rit == ind.rend())
                    break;
                ++(*rit);
                auto it = rit.base();
                while (it != ind.end()) {
                    *it = *rit;
                    ++it;
                }
            }
        }

        bool transform_ind (indices_t & ind) const {
            return std::next_permutation(ind.begin(), ind.end());
        }

        size_t number_of_equivalents (indices_t const& ind) const {
            return combinatorics::number_of_permutations(ind);
        }
    };

}
}
