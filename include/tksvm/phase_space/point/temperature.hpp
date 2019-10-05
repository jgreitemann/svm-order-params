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

#include <tksvm/phase_space/point/common.hpp>


namespace tksvm {
namespace phase_space {
namespace point {

    struct temperature {
        static const size_t label_dim = 1;
        using iterator = double *;
        using const_iterator = double const *;

        static void define_parameters(alps::params & params, std::string prefix="") {
            params.define<double>(prefix + "temp", 1., "temperature");
        }

        static bool supplied(alps::params const& params, std::string prefix="") {
            return params.supplied(prefix + "temp");
        }

        temperature() : temp(-1) {}
        temperature(double temp) : temp(temp) {}
        temperature(alps::params const& params, std::string prefix="")
            : temp(params[prefix + "temp"].as<double>()) {}

        template <class Iterator>
        temperature(Iterator begin) : temp(*begin) {}

        const_iterator begin() const { return &temp; }
        iterator begin() { return &temp; }
        const_iterator end() const { return &temp + 1; }
        iterator end() { return &temp + 1; }

        double temp;
    };

}
}
}
