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
    struct single {
        using ElementPolicy = BaseElementPolicy;
        using site_const_iterator = typename Container::const_iterator;

        struct unitcell;
        struct const_iterator {
            const_iterator & operator++ () { ++sit; return *this; }
            const_iterator operator++ (int) {
                const_iterator old(*this);
                ++(*this);
                return old;
            }
            const_iterator & operator-- () { --sit; return *this; }
            const_iterator operator-- (int) {
                const_iterator old(*this);
                --(*this);
                return old;
            }
            friend bool operator== (const_iterator lhs, const_iterator rhs) { return lhs.sit == rhs.sit; }
            friend bool operator!= (const_iterator lhs, const_iterator rhs) { return lhs.sit != rhs.sit; }
            unitcell operator* () const { return {sit}; }
            std::unique_ptr<unitcell> operator-> () const {
                return std::unique_ptr<unitcell>(new unitcell(sit));
            }
            friend single;
        private:
            const_iterator (site_const_iterator it) : sit {it} {}
            site_const_iterator sit;
        };

        struct unitcell {
            auto operator[](size_t block) const {
                typename Container::const_reference mat = *it;
                return mat.row(ElementPolicy::color_of_block(block));
            }
            friend const_iterator;
        private:
            unitcell (site_const_iterator it) : it {it} {}
            site_const_iterator it;
        };

        single (ElementPolicy, Container const& linear) : linear(linear) {}

        const_iterator begin () const {
            return {linear.begin()};
        }

        const_iterator end () const {
            return {linear.end()};
        }

        size_t size () const {
            return linear.size();
        }

    private:
        Container const& linear;
    };

}
}
}
