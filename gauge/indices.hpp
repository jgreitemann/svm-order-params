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

#include <iostream>
#include <vector>
#include <sstream>


using indices_t = std::vector<size_t>;

template <char first_letter>
struct basic_indices_t {
    indices_t ind;
    size_t & operator[] (size_t i) { return ind[i]; }
    size_t const& operator[] (size_t i) const { return ind[i]; }
    friend std::ostream & operator << (std::ostream & os,
                                       basic_indices_t const& indices)
        {
            for (size_t i : indices.ind)
                os << static_cast<char>(first_letter + i);
            return os;
        }
};

using block_indices_t = basic_indices_t<'l'>;
using contraction_indices_t = basic_indices_t<'a'>;
using component_indices_t = basic_indices_t<'x'>;

inline std::string block_str (indices_t const& bii, indices_t const& bjj) {
    std::stringstream ss;
    ss << '[' << block_indices_t{bii}
    << ';' << block_indices_t{bjj} << ']';
    return ss.str();
}
