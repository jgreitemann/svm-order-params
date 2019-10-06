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

#include <random>
#include <string>

#include <alps/params.hpp>
#include <alps/hdf5.hpp>

#include <tksvm/phase_space/sweep/policy.hpp>


namespace tksvm {
namespace phase_space {
namespace sweep {

    template <typename Point>
    struct uniform : public policy<Point> {
        using point_type = typename policy<Point>::point_type;
        using rng_type = typename policy<Point>::rng_type;

        static void define_parameters(alps::params & params,
                                      std::string const& prefix)
        {
            point_type::define_parameters(params, prefix + "uniform.a.");
            point_type::define_parameters(params, prefix + "uniform.b.");
            params.define<size_t>(prefix + "uniform.N", 10,
                "number of uniform point to draw");
        }

        uniform (size_t N, point_type a, point_type b)
            : N(N), a(a), b(b) {}

        uniform (alps::params const& params, std::string const& prefix)
            : N(params[prefix + "uniform.N"].as<size_t>())
            , a(params, prefix + "uniform.a.")
            , b(params, prefix + "uniform.b.") {}

        virtual size_t size() const final override {
            return N;
        }

        virtual bool yield (point_type & point, rng_type & rng) final override {
            auto ita = a.begin();
            auto itb = b.begin();
            for (auto & c : point) {
                c = std::uniform_real_distribution<double>{*ita, *itb}(rng);
                ++ita, ++itb;
            }
            return true;
        }

    private:
        size_t N;
        point_type a, b;
    };

}
}
}
