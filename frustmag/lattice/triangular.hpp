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

#include "lattice/bravais.hpp"

#include <utility>

namespace lattice {

template <typename Site>
struct triangular : bravais<Site, 2ul, 1ul> {
    using Base = bravais<Site, 2ul, 1ul>;
    using iterator = typename Base::iterator;
    using const_iterator = typename Base::const_iterator;

    static const size_t coordination = 6;

    using Base::bravais;

    auto nearest_neighbors(iterator const& it)
        -> std::array<iterator, coordination>
    {
        auto down = it.cell_it().down(0);
        auto up = it.cell_it().up(0);
        return {
            iterator{down, 0},
            iterator{up, 0},
            iterator{it.cell_it().down(1), 0},
            iterator{it.cell_it().up(1), 0},
            iterator{down.up(1), 0},
            iterator{up.down(1), 0}
        };
    }

    auto nearest_neighbors(const_iterator const& it) const
        -> std::array<const_iterator, coordination>
    {
        auto down = it.cell_it().down(0);
        auto up = it.cell_it().up(0);
        return {
            const_iterator{down, 0},
            const_iterator{up, 0},
            const_iterator{it.cell_it().down(1), 0},
            const_iterator{it.cell_it().up(1), 0},
            const_iterator{down.up(1), 0},
            const_iterator{up.down(1), 0}
        };
    }
};

}
