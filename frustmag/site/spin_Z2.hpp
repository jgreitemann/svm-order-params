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

#include <cstdint>
#include <random>

namespace site {

struct spin_Z2 {
    static const spin_Z2 up;
    static const spin_Z2 down;

    std::int8_t s;

    template <typename RNG>
    static spin_Z2 random(RNG & rng) {
        return {std::bernoulli_distribution{}(rng) ? up : down};
    }
    
    template <typename RNG>
    spin_Z2 flipped(RNG &) const {
        return { static_cast<std::int8_t>(-s) };
    }

    int dot(spin_Z2 other) const {
        return s * other.s;
    }

    operator std::int8_t() const {
        return s;
    }

    template <typename OutputIterator>
    OutputIterator & serialize(OutputIterator & it) const {
        *it = s;
        return ++it;
    }

    template <typename InputIterator>
    InputIterator & deserialize(InputIterator & it) {
        s = *it;
        return ++it;
    }
};

const spin_Z2 spin_Z2::up { static_cast<std::int8_t>(1) };
const spin_Z2 spin_Z2::down { static_cast<std::int8_t>(-1) };

static_assert(is_serializable<spin_Z2>::value, "spin_Z2 is not serializable");
static_assert(!is_archivable<spin_Z2>::value,
              "spin_Z2 is archivable, but shouldn't be");

#ifdef USE_CONCEPTS
static_assert(SiteState<spin_Z2>, "spin_Z2 is not a SiteState");
#endif

}
