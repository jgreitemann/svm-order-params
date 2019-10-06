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

#include <array>
#include <memory>
#include <stdexcept>

#include <combinatorics/ipow.hpp>

#include <tksvm/gauge/element_policy/n_partite.hpp>


namespace tksvm {
namespace gauge {
namespace cluster_policy {

    template <typename BaseElementPolicy, typename Container, size_t DIM = 3>
    struct square {
        using ElementPolicy = element_policy::n_partite<BaseElementPolicy>;
        using site_const_iterator = typename Container::const_iterator;
        using coord_t = std::array<size_t, DIM>;

        struct unitcell;
        struct const_iterator {
            const_iterator & operator++ () {
                coord[0] += 2;
                if (coord[0] >= L) {
                    size_t i;
                    for (i = 1; i < coord.size(); ++i) {
                        ++coord[i];
                        if (coord[i] < L)
                            break;
                        coord[i] = 0;
                    }
                    if (i == coord.size()) {
                        coord[0] = L;
                    } else {
                        size_t sum = 0;
                        for (i = 1; i < coord.size(); ++i) {
                            sum += coord[i];
                        }
                        coord[0] = sum % 2;
                    }
                }
                return *this;
            }
            const_iterator operator++ (int) {
                const_iterator old(*this);
                ++(*this);
                return old;
            }
            friend bool operator== (const_iterator lhs, const_iterator rhs) {
                return (lhs.coord == rhs.coord
                        && lhs.root == rhs.root
                        && lhs.L == rhs.L);
            }
            friend bool operator!= (const_iterator lhs, const_iterator rhs) { return !(lhs == rhs); }
            unitcell operator* () const { return {root, lin_index(), L}; }
            std::unique_ptr<unitcell> operator-> () const {
                return std::make_unique<unitcell>(root, lin_index(), L);
            }
            friend square;
        private:
            const_iterator (site_const_iterator it, coord_t c, size_t L)
                : root{it}, coord{c}, L{L} {}
            size_t lin_index () const {
                size_t sum = 0;
                for (size_t c : coord) {
                    sum *= L;
                    sum += c;
                }
                return sum;
            }
            site_const_iterator root;
            coord_t coord;
            size_t L;
        };

        struct unitcell {
            auto operator[](size_t block) const {
                size_t subl = ElementPolicy::sublattice_of_block(block);
                size_t color = ElementPolicy::color_of_block(block);
                typename Container::const_reference mat =
                    root[subl ? (idx / L * L + (idx + 1) % L) : idx];
                return mat.row(color);
            }
            friend const_iterator;
        private:
            unitcell (site_const_iterator it, size_t idx, size_t L)
                : root{it}, idx{idx}, L{L} {}
            site_const_iterator root;
            size_t idx;
            size_t L;
        };

        square (ElementPolicy, Container const& linear) : linear(linear) {
            L = static_cast<size_t>(pow(linear.size() + 0.5, 1./DIM));
            if (combinatorics::ipow(L, DIM) != linear.size())
                throw std::runtime_error("linear configuration size doesn't match DIM");
            if (L % 2 != 0)
                throw std::runtime_error("lattice not bipartite w.r.t. PBCs");
        }

        const_iterator begin () const {
            return {linear.begin(), {0}, L};
        }

        const_iterator end () const {
            return {linear.begin(), {L}, L};
        }

        size_t size () const {
            return linear.size() / 2;
        }

    private:
        Container const& linear;
        size_t L;
    };

}
}
}
