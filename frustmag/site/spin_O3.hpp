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

#include <Eigen/Dense>

#include <cmath>
#include <random>

namespace site {

struct spin_O3 : Eigen::Vector3d {
    template <typename RNG>
    static spin_O3 random(RNG & rng) {
        double phi = std::uniform_real_distribution<double>{0, 2*M_PI}(rng);
        double cos_theta = std::uniform_real_distribution<double>{-1, 1}(rng);
        double sin_theta = std::sqrt(1 - std::pow(cos_theta, 2));
        spin_O3 ret;
        ret <<
            sin_theta * std::cos(phi),
            sin_theta * std::sin(phi),
            cos_theta;
        return ret;
    }
    
    template <typename RNG>
    spin_O3 flipped(RNG & rng) const {
        return random(rng);
    }
};

#ifdef USE_CONCEPTS
static_assert(SiteState<spin_O3>, "spin_O3 is not a SiteState");
#endif

}
