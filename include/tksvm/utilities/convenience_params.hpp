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

#include <alps/params.hpp>
#include "filesystem.hpp"

inline alps::params& define_convenience_parameters(alps::params & parameters) {
    const std::string origin = alps::origin_name(parameters);
    parameters
        .define<std::size_t>("timelimit", 0, "time limit for the simulation")
        .define<std::string>("outputfile",
                             replace_extension(origin, ".out.h5"),
                             "name of the output file")
        .define<std::string>("checkpoint",
                             replace_extension(origin, ".clone.h5"),
                             "name of the checkpoint file to save to")
        ;
    return parameters;
}
