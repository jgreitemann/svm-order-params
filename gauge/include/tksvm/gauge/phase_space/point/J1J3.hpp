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

#include <string>

#include <alps/params.hpp>

#include <tksvm/phase_space/classifier/phase_diagram.hpp>
#include <tksvm/phase_space/point/common.hpp>


namespace tksvm {
namespace phase_space {
namespace point {

    struct J1J3 {
        static const size_t label_dim = 2;
        using iterator = double *;
        using const_iterator = double const *;

        static void define_parameters(alps::params & params, std::string prefix="") {
            params
                .define<double>(prefix + "J1", 0., "J1 coupling")
                .define<double>(prefix + "J3", 0., "J3 coupling");
        }

        static bool supplied(alps::params const& params, std::string prefix="") {
            return params.supplied(prefix + "J1")
                && params.supplied(prefix + "J3");
        }

        J1J3(alps::params const& params, std::string prefix="")
            : J{params[prefix + "J1"].as<double>(),
                params[prefix + "J3"].as<double>()} {}

        J1J3() : J{-1, -1} {}
        J1J3(double J1, double J3) : J{J1, J3} {}

        template <class Iterator>
        J1J3(Iterator begin) : J {*begin, *(++begin)} {}

        const_iterator begin() const { return J; }
        iterator begin() { return J; }
        const_iterator end() const { return J + 2; }
        iterator end() { return J + 2; }

        double const& J1() const { return J[0]; }
        double & J1() { return J[0]; }
        double const& J3() const { return J[1]; }
        double & J3() { return J[1]; }

        double J[2];
    };

}

namespace classifier {

    template <>
    struct phase_diagram_database<point::J1J3> {
        static const typename phase_diagram<point::J1J3>::map_type map;
    };

}
}
}
