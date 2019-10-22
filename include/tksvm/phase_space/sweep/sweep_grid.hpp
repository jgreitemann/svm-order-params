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
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <alps/params.hpp>
#include <alps/hdf5.hpp>

#include <tksvm/phase_space/sweep/policy.hpp>


namespace tksvm {
namespace phase_space {
namespace sweep {

    template <typename Point>
    struct sweep_grid : public policy<Point> {
        using point_type = typename policy<Point>::point_type;
        using rng_type = typename policy<Point>::rng_type;

        static const size_t dim = point_type::label_dim;

        static void define_parameters(alps::params & params,
                                      std::string const& prefix)
        {
            for (size_t i = 1; i <= dim; ++i) {
                sweep::define_parameters<point_type>(params,
                    format_sweep(i, prefix), false);
            }
        }

        sweep_grid(alps::params const& params,
                   std::string const& prefix)
        {
            for (size_t i = 0; i < dim; ++i) {
                sweeps[i] = sweep::from_parameters<point_type>(params,
                    format_sweep(i + 1, prefix));
                ns[i] = sweeps[i]->size() - 1;
            }
        }

        virtual size_t size() const final override {
            return std::accumulate(sweeps.begin(), sweeps.end(), 1ul,
                [](size_t acc, sweep_ptr const& s) {
                    return acc * s->size();
                });
        }

        virtual bool yield (point_type & point, rng_type & rng) final override {
            for (size_t i = 0; i < dim; ++i) {
                sweeps[i]->yield(points[i], rng);
                ++ns[i];
                if (ns[i] == sweeps[i]->size()) {
                    ns[i] = 0;
                } else {
                    break;
                }
            }

            size_t i = 0;
            for (double & x : point) {
                x = *std::next(points[i].begin(), i);
                ++i;
            }
            return true;
        }

        virtual void save (alps::hdf5::archive & ar) const final override {
            ar["ns"] << std::vector<size_t>{ns.begin(), ns.end()};
        }

        virtual void load (alps::hdf5::archive & ar) final override {
            std::vector<size_t> ns_vec;
            ar["ns"] >> ns_vec;
            std::copy(ns_vec.begin(), ns_vec.end(), ns.begin());
        }

        static std::string format_sweep(size_t i,
                                        std::string const& prefix)
        {
            std::stringstream ss;
            ss << prefix + "sweep_grid.sweep" << i << '.';
            return ss.str();
        }

    private:
        using sweep_ptr = std::unique_ptr<policy<point_type>>;
        std::array<sweep_ptr, dim> sweeps;
        std::array<point_type, dim> points;
        std::array<size_t, dim> ns;
    };

}
}
}
