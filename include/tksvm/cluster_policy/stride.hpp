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

#include <iterator>
#include <memory>


namespace tksvm {
namespace cluster_policy {

    template <typename BaseElementPolicy, typename Container>
    struct stride {
        using ElementPolicy = BaseElementPolicy;
        using site_const_iterator = typename Container::const_iterator;

        struct unitcell;
        struct const_iterator {
            const_iterator & operator++ () { sit += range_; return *this; }
            const_iterator operator++ (int) {
                const_iterator old(*this);
                ++(*this);
                return old;
            }
            const_iterator & operator-- () { sit -= range_; return *this; }
            const_iterator operator-- (int) {
                const_iterator old(*this);
                --(*this);
                return old;
            }
            friend bool operator== (const_iterator lhs, const_iterator rhs) { return lhs.sit == rhs.sit; }
            friend bool operator!= (const_iterator lhs, const_iterator rhs) { return lhs.sit != rhs.sit; }
            unitcell operator* () const { return {sit, stride_}; }
            std::unique_ptr<unitcell> operator-> () const {
                return std::unique_ptr<unitcell>(new unitcell(sit, stride_));
            }
            friend stride;
        private:
            const_iterator (site_const_iterator it, size_t range, size_t stride)
                : sit {it}, range_{range}, stride_{stride} {}
            site_const_iterator sit;
            size_t range_, stride_;
        };

        struct unitcell {
            auto operator[](size_t block) const {
                return it + block * stride_;
            }
            friend const_iterator;
        private:
            unitcell (site_const_iterator it, size_t stride)
                : it{it}, stride_{stride} {}
            site_const_iterator it;
            size_t stride_;
        };

        stride(ElementPolicy && elempol, Container const& linear)
            : linear{linear}
            , range_{elempol.range()}
            , stride_{elempol.range() / elempol.n_block()}
        {
        }

        const_iterator begin () const {
            return {linear.begin(), range_, stride_};
        }

        const_iterator end () const {
            return {linear.end(), range_, stride_};
        }

        size_t size () const {
            return std::distance(linear.begin(), linear.end()) / range_;
        }

    private:
        Container const& linear;
        size_t range_, stride_;
    };

}
}
