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

#include <cstdlib>


namespace tksvm {
namespace gauge {
namespace element_policy {

    struct triad {
        constexpr size_t n_block() const { return 3; }
        constexpr size_t range() const { return 9; }
        constexpr size_t block(size_t index) const { return index / 3; }
        constexpr size_t component(size_t index) const { return index % 3; }
        static constexpr size_t n_color() { return 3; }
        static constexpr size_t sublattice_of_block(size_t) { return 0; }
        static constexpr size_t color_of_block(size_t block) { return block; }
    };

}
}
}
