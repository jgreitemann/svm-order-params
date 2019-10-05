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

#include <map>
#include <utility>
#include <vector>

#include <boost/multi_array.hpp>

#include <tksvm/utilities/indices.hpp>


namespace tksvm {
namespace config {

template <typename Config, typename Introspector>
struct policy {
    using config_array = Config;
    using introspec_t = Introspector;

    using matrix_t = boost::multi_array<double, 2>;

    virtual size_t size () const = 0;
    virtual size_t range () const = 0;
    virtual size_t n_components () const = 0;
    virtual size_t rank () const = 0;
    virtual std::vector<double> configuration (config_array const&) const = 0;

    virtual matrix_t rearrange (matrix_t const& c) const = 0;
    virtual matrix_t rearrange (introspec_t const& c,
                                indices_t const& bi,
                                indices_t const& bj) const = 0;
    virtual std::pair<matrix_t, matrix_t> block_structure (matrix_t const& c) const = 0;

    virtual indices_t block_indices(indices_t const& ind) const = 0;
    virtual indices_t component_indices(indices_t const& ind) const = 0;

    virtual std::map<indices_t, index_assoc_vec> all_block_indices () const = 0;
};

}
}
