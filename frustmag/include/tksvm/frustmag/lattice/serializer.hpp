// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2019  Jonas Greitemann, Ke Liu, and Lode Pollet

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


namespace tksvm {
namespace frustmag {
namespace lattice {

template <typename Lattice>
struct serializer {
    using config_t = Lattice;
    template <typename OutputIterator>
    void serialize(config_t const& lattice, OutputIterator it) const {
    	for (auto const& site : lattice)
    		site.serialize(it);
    }

    template <typename InputIterator>
    void deserialize(InputIterator it, config_t & lattice) const {
    	for (auto & site : lattice)
    		site.deserialize(it);
    }
};

}
}
}
