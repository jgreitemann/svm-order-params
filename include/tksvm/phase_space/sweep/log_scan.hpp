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
#include <string>

#include <alps/params.hpp>
#include <alps/hdf5.hpp>

#include <tksvm/phase_space/sweep/policy.hpp>


namespace tksvm {
namespace phase_space {
namespace sweep {

    template <typename Point>
    struct log_scan : public policy<Point> {
        using point_type = Point;
        using rng_type = typename policy<Point>::rng_type;

        static void define_parameters(alps::params & params,
                                      std::string const& prefix)
        {
            point_type::define_parameters(params, prefix + "log_scan.a.");
            point_type::define_parameters(params, prefix + "log_scan.b.");
            for (size_t i = 0; i < point_type::label_dim; ++i) {
                params.define<bool>(
                    prefix + "log_scan.log" + std::to_string(i + 1), true,
                    "boolean indicating if axis" + std::to_string(i + 1)
                    + " is logarithmic");
            }
            params.define<size_t>(prefix + "log_scan.N", 8,
                "number of subdivisions");
        }

        log_scan (alps::params const& params,
                  std::string const& prefix,
                  size_t offset = 0)
            : a(params, prefix + "log_scan.a.")
            , b(params, prefix + "log_scan.b.")
            , N(params[prefix + "log_scan.N"].as<size_t>())
            , n(offset)
        {
            for (size_t i = 0; i < point_type::label_dim; ++i) {
                is_log[i] = params[prefix + "log_scan.log"
                    + std::to_string(i + 1)].as<bool>();
            }
        }

        size_t size() const final override {
            return N;
        }

        virtual bool yield (point_type & point, rng_type &) final override {
            return yield(point);
        }

        bool yield (point_type & point) {
            auto it_a = a.begin();
            auto it_b = b.begin();
            auto it = point.begin();
            for (size_t i = 0; i < point_type::label_dim;
                 ++i, ++it_a, ++it_b, ++it)
            {
                double x = (N == 1) ? 0 : 1. * n / (N - 1);
                if (is_log[i])
                    *it = *it_a * pow(*it_b / *it_a, x);
                else
                    *it = *it_a + (*it_b - *it_a) * x;
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
        std::array<bool, point_type::label_dim> is_log;
        size_t N;
        size_t n;
    };

}
}
}
