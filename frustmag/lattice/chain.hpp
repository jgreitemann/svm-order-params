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

#include <alps/params.hpp>

#include <algorithm>
#include <array>
#include <iterator>
#include <vector>

namespace lattice {

template <typename Site>
struct chain : std::vector<Site> {
    using Base = std::vector<Site>;
    using value_type = Site;
    using iterator = typename Base::iterator;
    using const_iterator = typename Base::const_iterator;

    using std::vector<Site>::vector;

    static void define_parameters(alps::params & parameters) {
        parameters
            .define<size_t>("lattice.chain.length", "length of the chain")
            .define<bool>("lattice.chain.periodic", 1, "PBC = true, OBC = false");
    }

    template <typename Generator>
    chain(alps::params const& parameters, Generator && gen) {
        size_t L = parameters["lattice.chain.length"];
        this->reserve(L);
        std::generate_n(std::back_inserter(*this), L, gen);
    }

    auto nearest_neighbors(iterator it) -> std::array<iterator, 2> {
        if (periodic) {
            return {
                it == this->begin() ? std::prev(this->end()) : std::prev(it),
                ++it == this->end() ? this->begin() : it
            };
        } else {
            return {
                it == this->begin() ? this->end() : std::prev(it),
                ++it
            };
        }
    }

    auto nearest_neighbors(const_iterator it) const
        -> std::array<const_iterator, 2>
    {
        if (periodic) {
            return {
                it == this->begin() ? std::prev(this->end()) : std::prev(it),
                ++it == this->end() ? this->begin() : it
            };
        } else {
            return {
                it == this->begin() ? this->end() : std::prev(it),
                ++it
            };
        }
    }

private:
    bool periodic;
};

}
