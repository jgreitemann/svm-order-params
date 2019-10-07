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

#include <algorithm>
#include <string>


namespace tksvm {

inline std::string replace_extension(std::string const& old_path,
                                     std::string const& new_ext)
{
    auto slash_pos = old_path.find_last_of('.');
    auto dot_pos = old_path.find_first_of('.', slash_pos == std::string::npos ? 0 : slash_pos);
    return (dot_pos == std::string::npos ? old_path : old_path.substr(0, dot_pos)) + new_ext;
}

}
