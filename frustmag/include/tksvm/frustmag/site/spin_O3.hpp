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

#include <cmath>
#include <random>

#include <Eigen/Dense>

#include <tksvm/frustmag/concepts.hpp>


namespace tksvm {
namespace frustmag {
namespace site {

struct spin_O3 : Eigen::Vector3d {
    template <typename RNG>
    static spin_O3 random(RNG & rng, double cos_theta_0 = -1) {
        double phi = std::uniform_real_distribution<double>{0, 2*M_PI}(rng);
        double cos_theta = std::uniform_real_distribution<double>{cos_theta_0, 1}(rng);
        double sin_theta = std::sqrt(1 - std::pow(cos_theta, 2));
        spin_O3 ret;
        ret <<
            sin_theta * std::cos(phi),
            sin_theta * std::sin(phi),
            cos_theta;
        return ret;
    }

    spin_O3() = default;

    spin_O3(Eigen::Vector3d const& other) : Eigen::Vector3d{other} {}

    spin_O3 relative_spin(spin_O3 const& other) const {
        double norm = this->norm();
        double cos_theta = (*this)[2] / norm;
        double sin_theta = std::sqrt(1 - std::pow(cos_theta, 2));
        double cos_phi = (*this)[0] / norm / sin_theta;
        double sin_phi = (*this)[1] / norm / sin_theta;
        Eigen::Matrix3d R;
        R <<
            cos_theta * cos_phi, -sin_phi, sin_theta * cos_phi,
            cos_theta * sin_phi, cos_phi, sin_theta * sin_phi,
            -sin_theta, 0, cos_theta;
        spin_O3 ret{R * other};

        // important: renormalize to avoid exponential growth of rounding errors
        ret /= ret.norm();

        return ret;
    }

    template <typename RNG>
    spin_O3 flipped(RNG & rng, double cos_theta_0 = -1) const {
        return relative_spin(random(rng, cos_theta_0));
    }

    static const size_t size = 3;

    template <typename OutputIterator>
    OutputIterator & serialize(OutputIterator & it) const {
        *it = (*this)(0);
        *(++it) = (*this)(1);
        *(++it) = (*this)(2);
        return ++it;
    }

    template <typename InputIterator>
    InputIterator & deserialize(InputIterator & it) {
        (*this)(0) = *it;
        (*this)(1) = *(++it);
        (*this)(2) = *(++it);
        return ++it;
    }
};

static_assert(is_serializable<spin_O3>::value, "spin_O3 is not serializable");
static_assert(!is_archivable<spin_O3>::value,
              "spin_O3 is archivable, but shouldn't be");

#ifdef USE_CONCEPTS
static_assert(SiteState<spin_O3>, "spin_O3 is not a SiteState");
#endif

}
}
}
