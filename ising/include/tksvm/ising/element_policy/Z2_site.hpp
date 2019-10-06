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
namespace ising {
namespace element_policy {

    struct Z2_site {
        Z2_site(size_t L) : L(L) {}
        constexpr size_t n_block() const { return L; }
        constexpr size_t range() const { return L; }
        constexpr size_t block(size_t index) const { return index; }
        constexpr size_t component(size_t) const { return 0; }

    private:
        size_t L;
    };

}
}
}
