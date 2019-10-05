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
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

#include <alps/params.hpp>
#include <alps/hdf5.hpp>

#include <tksvm/phase_space/sweep/policy.hpp>


namespace tksvm {
namespace phase_space {
namespace sweep {

    template <typename Point>
    struct grid : public policy<Point> {
        using point_type = typename policy<Point>::point_type;
        using rng_type = typename policy<Point>::rng_type;

        static const size_t dim = point_type::label_dim;

        static void define_parameters(alps::params & params,
                                      std::string const& prefix)
        {
            point_type::define_parameters(params, prefix + "grid.a.");
            point_type::define_parameters(params, prefix + "grid.b.");
            for (size_t i = 1; i <= dim; ++i)
                params.define<size_t>(format_subdiv(i, prefix), 1,
                                      "number of grid subdivisions");
        }

        grid(alps::params const& params,
             std::string const& prefix,
             size_t offset = 0)
            : a(params, prefix + "grid.a.")
            , b(params, prefix + "grid.b.")
            , n(offset)
        {
            for (size_t i = 1; i <= dim; ++i)
                subdivs[i-1] = params[format_subdiv(i, prefix)].as<size_t>();

            double frac = 0.;
            for (double p = 0.5; offset != 0; offset >>= 1, p /= 2)
                if (offset & 1)
                    frac += p;
            n = size() * frac;
        }

        virtual size_t size() const final override {
            return std::accumulate(subdivs.begin(), subdivs.end(),
                                   1, std::multiplies<>{});
        }

        bool yield (point_type & point) {
            size_t x = n;
            auto it = point.begin();
            auto ita = a.begin();
            auto itb = b.begin();
            for (size_t i = 0; i < dim; ++i, ++it, ++ita, ++itb) {
                if (subdivs[i] == 1) {
                    *it = *ita;
                } else {
                    *it = *ita + (*itb - *ita) / (subdivs[i] - 1) * (x % subdivs[i]);
                    x /= subdivs[i];
                }
            }
            n = (n + 1) % size();
            return true;
        }

        virtual bool yield (point_type & point, rng_type &) final override {
            return yield(point);
        }

        virtual void save (alps::hdf5::archive & ar) const final override {
            ar["n"] << n;
        }

        virtual void load (alps::hdf5::archive & ar) final override {
            ar["n"] >> n;
        }

        static std::string format_subdiv(size_t i,
                                         std::string const& prefix)
        {
            std::stringstream ss;
            ss << prefix + "grid.N" << i;
            return ss.str();
        }

    private:
        std::array<size_t, dim> subdivs;
        point_type a, b;
        size_t n;
    };

}
}
}
