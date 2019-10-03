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

template <typename Site, size_t dim>
struct ortho : bravais<Site, dim, 1ul> {
    using Base = bravais<Site, dim, 1ul>;
    using iterator = typename Base::iterator;
    using const_iterator = typename Base::const_iterator;

    static const size_t coordination = 2 * dim;

    using Base::bravais;

    auto nearest_neighbors(iterator const& it)
        -> std::array<iterator, coordination>
    {
        using Indices = std::make_index_sequence<dim>;
        return nearest_neighbor_impl(it, Indices{});
    }

    auto nearest_neighbors(const_iterator const& it) const
        -> std::array<const_iterator, coordination>
    {
        using Indices = std::make_index_sequence<dim>;
        return nearest_neighbor_impl(it, Indices{});
    }

private:
    template <typename It, size_t... I>
    auto nearest_neighbor_impl(It const& it, std::index_sequence<I...>) const
        -> std::array<It, coordination>
    {
        return {
            It{it.cell_it().down(I), 0}...,
            It{it.cell_it().up(I), 0}...
        };
    }

};

// template <typename Site>
// using chain = ortho<Site, 1>;

template <typename Site>
using square = ortho<Site, 2>;

template <typename Site>
using cubic = ortho<Site, 3>;

}

template <typename Site, size_t dim>
struct config_serializer<lattice::ortho<Site, dim>>
    : lattice_serializer<lattice::ortho<Site, dim>> {};
