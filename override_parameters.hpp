// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2017  Jonas Greitemann, Ke Liu, and Lode Pollet

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

#include "argh.h"


template <typename T>
bool override_parameter (std::string const& name, alps::params & parameters, argh::parser & cmdl) {
    T new_param;
    if (cmdl(name) >> new_param) {
        std::cout << "override parameter " << name << ": " << new_param << std::endl;
        parameters[name] = new_param;
        return true;
    }
    return false;
}
