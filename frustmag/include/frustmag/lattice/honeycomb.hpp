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
struct honeycomb : bravais<Site, 2ul, 2ul> {
    using Base = bravais<Site, 2ul, 2ul>;
    using iterator = typename Base::iterator;
    using const_iterator = typename Base::const_iterator;

    static const size_t coordination = 3;

    using Base::bravais;

    auto nearest_neighbors(iterator const& it)
        -> std::array<iterator, coordination>
    {
        if (it.basis_index() == 0)
            return {
                iterator{it.cell_it(), 1},
                iterator{it.cell_it().down(0), 1},
                iterator{it.cell_it().down(1), 1},
            };
        else
            return {
                iterator{it.cell_it(), 0},
                iterator{it.cell_it().up(0), 0},
                iterator{it.cell_it().up(1), 0},
            };
    }

    auto nearest_neighbors(const_iterator const& it) const
        -> std::array<const_iterator, coordination>
    {
        if (it.basis_index() == 0)
            return {
                const_iterator{it.cell_it(), 1},
                const_iterator{it.cell_it().down(0), 1},
                const_iterator{it.cell_it().down(1), 1},
            };
        else
            return {
                const_iterator{it.cell_it(), 0},
                const_iterator{it.cell_it().up(0), 0},
                const_iterator{it.cell_it().up(1), 0},
            };
    }
};

}

template <typename Site>
struct config_serializer<lattice::honeycomb<Site>>
    : lattice_serializer<lattice::honeycomb<Site>> {};
