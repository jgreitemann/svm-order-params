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
#include <random>

#include <alps/hdf5.hpp>
#include <alps/params.hpp>


namespace tksvm {
namespace phase_space {
namespace sweep {

    template <typename Point, typename RNG = std::mt19937>
    struct policy {
        using point_type = Point;
        using rng_type = RNG;

        virtual ~policy() noexcept = default;

        virtual size_t size() const = 0;
        virtual bool yield (point_type & point, rng_type & rng) = 0;
        virtual void save (alps::hdf5::archive &) const {}
        virtual void load (alps::hdf5::archive &) {}
    };

    template <typename Point>
    void define_parameters(alps::params &, std::string const&, bool = true);

    template <typename Point>
    auto from_parameters(alps::params const&, std::string const&, size_t = 0)
        -> std::unique_ptr<policy<Point>>;

}
}
}
