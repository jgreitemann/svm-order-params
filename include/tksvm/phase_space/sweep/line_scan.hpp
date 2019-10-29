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

#include <string>

#include <alps/params.hpp>
#include <alps/hdf5.hpp>

#include <tksvm/phase_space/sweep/policy.hpp>


namespace tksvm {
namespace phase_space {
namespace sweep {

    template <typename Point>
    struct line_scan : public policy<Point> {
        using point_type = typename policy<Point>::point_type;
        using rng_type = typename policy<Point>::rng_type;

        static void define_parameters(alps::params & params,
                                      std::string const& prefix)
            {
                point_type::define_parameters(params, prefix + "line_scan.a.");
                point_type::define_parameters(params, prefix + "line_scan.b.");
                params.define<size_t>(prefix + "line_scan.N", 8,
                                      "number of phase points on line");
            }

        line_scan (alps::params const& params,
                   std::string const& prefix,
                   size_t offset = 0)
            : a(params, prefix + "line_scan.a.")
            , b(params, prefix + "line_scan.b.")
            , n(offset)
            , N(params[prefix + "line_scan.N"].as<size_t>())
        {
        }

        line_scan (point_type const& a, point_type const& b, size_t N)
            : a(a), b(b), n(0), N(N) {}

        virtual size_t size() const final override {
            return N;
        }

        virtual bool yield (point_type & point, rng_type &) final override {
            return yield(point);
        }

        bool yield (point_type & point) {
            auto it_a = a.begin();
            auto it_b = b.begin();
            double x = (N == 1) ? 0. : 1. * n / (N - 1);
            for (auto & c : point) {
                c = *it_b * x + *it_a * (1. - x);
                ++it_a, ++it_b;
            }
            n = (n + 1) % size();
            return true;
        }

        virtual void save (alps::hdf5::archive & ar) const final override {
            ar["n"] << n;
        }

        virtual void load (alps::hdf5::archive & ar) final override {
            ar["n"] >> n;
        }

    private:
        const point_type a, b;
        size_t n, N;
    };

}
}
}
