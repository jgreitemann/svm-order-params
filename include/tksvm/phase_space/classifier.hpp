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

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <alps/params.hpp>

#include <tksvm/phase_space/classifier/critical_temperature.hpp>
#include <tksvm/phase_space/classifier/fixed_from_sweep.hpp>
#include <tksvm/phase_space/classifier/hyperplane.hpp>
#include <tksvm/phase_space/classifier/orthants.hpp>
#include <tksvm/phase_space/classifier/phase_diagram.hpp>
#include <tksvm/phase_space/classifier/policy.hpp>


namespace tksvm {
namespace phase_space {
namespace classifier {

    template <typename Point>
    void define_parameters(alps::params & params,
                           std::string && prefix = "classifier.")
    {
        params.define<std::string>(prefix + "policy",
            "fixed_from_sweep",
            "name of the classifier policy");
        orthants<Point>::define_parameters(params, prefix);
        hyperplane<Point>::define_parameters(params, prefix);
        phase_diagram<Point>::define_parameters(params, prefix);
        fixed_from_sweep<Point>::define_parameters(params, prefix);
        if (std::is_same<Point, point::temperature>::value)
            critical_temperature::define_parameters(params, prefix);
    }

    template <typename Point>
    auto from_parameters(alps::params const& params,
                         std::string && prefix = "classifier.")
    {
        return std::unique_ptr<policy<Point>>{[&] () -> policy<Point>* {
            std::string pol_name = params[prefix + "policy"];
            if (pol_name == "orthants")
                return dynamic_cast<policy<Point>*>(
                    new orthants<Point>(params, prefix));
            if (pol_name == "hyperplane")
                return dynamic_cast<policy<Point>*>(
                    new hyperplane<Point>(params, prefix));
            if (pol_name == "phase_diagram")
                return dynamic_cast<policy<Point>*>(
                    new phase_diagram<Point>(params, prefix));
            if (pol_name == "fixed_from_sweep")
                return dynamic_cast<policy<Point>*>(
                    new fixed_from_sweep<Point>(params, prefix));
            if (pol_name == "critical_temperature") {
                if (std::is_same<Point, point::temperature>::value)
                    return dynamic_cast<policy<Point>*>(
                        new critical_temperature(params, prefix));
                else
                    throw std::runtime_error(
                        "critical_temperature classifier is only available "
                        "for use with `temperature` phase space points");
            }
            throw std::runtime_error("Invalid classifier policy \""
                + pol_name + "\"");
            return nullptr;
        }()};
    }

}
}
}
