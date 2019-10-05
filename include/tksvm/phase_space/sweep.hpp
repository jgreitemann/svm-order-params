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

#include <alps/params.hpp>

#include <tksvm/phase_space/sweep/cycle.hpp>
#include <tksvm/phase_space/sweep/grid.hpp>
#include <tksvm/phase_space/sweep/line_scan.hpp>
#include <tksvm/phase_space/sweep/log_scan.hpp>
#include <tksvm/phase_space/sweep/nonuniform_grid.hpp>
#include <tksvm/phase_space/sweep/policy.hpp>
#include <tksvm/phase_space/sweep/sweep_grid.hpp>
#include <tksvm/phase_space/sweep/uniform.hpp>
#include <tksvm/phase_space/sweep/uniform_line.hpp>


namespace tksvm {
namespace phase_space {
namespace sweep {

    template <typename Point>
    void define_parameters(alps::params & params,
                           std::string const& prefix,
                           bool allow_recursive)
    {
        params.define<std::string>(prefix + "policy", "cycle",
           "phase space point sweep policy name");
        cycle<Point>::define_parameters(params, prefix);
        grid<Point>::define_parameters(params, prefix);
        nonuniform_grid<Point>::define_parameters(params, prefix);
        if (allow_recursive)
            sweep_grid<Point>::define_parameters(params, prefix);
        uniform<Point>::define_parameters(params, prefix);
        uniform_line<Point>::define_parameters(params, prefix);
        line_scan<Point>::define_parameters(params, prefix);
        log_scan<Point>::define_parameters(params, prefix);
    }

    template <typename Point>
    auto from_parameters(alps::params const& params,
                         std::string const& prefix,
                         size_t seed_offset)
        -> std::unique_ptr<policy<Point>>
    {
        return std::unique_ptr<policy<Point>>{[&] () -> policy<Point>* {
            std::string pol_name = params[prefix + "policy"];
            if (pol_name == "cycle")
                return dynamic_cast<policy<Point>*>(
                    new cycle<Point>(params, prefix, seed_offset));
            if (pol_name == "grid")
                return dynamic_cast<policy<Point>*>(
                    new grid<Point>(params, prefix, seed_offset));
            if (pol_name == "nonuniform_grid")
                return dynamic_cast<policy<Point>*>(
                    new nonuniform_grid<Point>(params, prefix, seed_offset));
            if (pol_name == "sweep_grid")
                return dynamic_cast<policy<Point>*>(
                    new sweep_grid<Point>(params, prefix, seed_offset));
            if (pol_name == "uniform")
                return dynamic_cast<policy<Point>*>(
                    new uniform<Point>(params, prefix));
            if (pol_name == "uniform_line")
                return dynamic_cast<policy<Point>*>(
                    new uniform_line<Point>(params, prefix));
            if (pol_name == "line_scan")
                return dynamic_cast<policy<Point>*>(
                    new line_scan<Point> (params, prefix, seed_offset));
            if (pol_name == "log_scan")
                return dynamic_cast<policy<Point>*>(
                    new log_scan<Point> (params, prefix, seed_offset));
            throw std::runtime_error("Invalid sweep policy \""
                + pol_name + "\"");
            return nullptr;
        }()};
    }

}
}
}
