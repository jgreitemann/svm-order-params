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

#include <tksvm/frustmag/element_policy/lattice.hpp>


namespace tksvm {
namespace frustmag {
namespace cluster_policy {

    template <typename Lattice>
    struct lattice {
        using ElementPolicy = element_policy::lattice<Lattice>;
        using const_iterator = typename Lattice::unitcell_collection_type::const_iterator;

        lattice(ElementPolicy, Lattice const& lat)
            : cells_{lat.cells()}
        {
        }

        const_iterator begin() const {
            return cells_.begin();
        }

        const_iterator end() const {
            return cells_.end();
        }

        auto size() const {
            return cells_.size();
        }

    private:
        typename Lattice::unitcell_collection_type const& cells_;
    };

}
}
}
