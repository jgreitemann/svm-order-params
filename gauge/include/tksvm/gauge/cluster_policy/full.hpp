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

#include <memory>

#include <tksvm/gauge/element_policy/n_partite.hpp>


namespace tksvm {
namespace gauge {
namespace cluster_policy {

    template <typename BaseElementPolicy, typename Container>
    struct full {
        using ElementPolicy = element_policy::n_partite<BaseElementPolicy>;
        using site_const_iterator = typename Container::const_iterator;

        struct unitcell;
        struct const_iterator {
            const_iterator & operator++ () {
                root += size;
                return *this;
            }
            const_iterator operator++ (int) {
                const_iterator old(*this);
                ++(*this);
                return old;
            }
            friend bool operator== (const_iterator lhs, const_iterator rhs) {
                return lhs.root == rhs.root;
            }
            friend bool operator!= (const_iterator lhs, const_iterator rhs) { return !(lhs == rhs); }
            unitcell operator* () const { return {root}; }
            std::unique_ptr<unitcell> operator-> () const {
                return std::make_unique<unitcell>(root);
            }
            site_const_iterator root;
            size_t size;
        };

        struct unitcell {
            auto operator[](size_t block) const {
                size_t subl = ElementPolicy::sublattice_of_block(block);
                size_t color = ElementPolicy::color_of_block(block);
                typename Container::const_reference mat = root[subl];
                return mat.row(color);
            }
            site_const_iterator root;
        };

        full (ElementPolicy, Container const& linear) : linear(linear) {}

        const_iterator begin () const {
            return {linear.begin(), linear.size()};
        }

        const_iterator end () const {
            return {linear.end(), linear.size()};
        }

        size_t size () const {
            return 1;
        }

    private:
        Container const& linear;
    };

}
}
}
