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

#include <algorithm>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <alps/params.hpp>

#include <tksvm/phase_space/sweep.hpp>
#include <tksvm/phase_space/classifier/policy.hpp>
#include <tksvm/phase_space/point/common.hpp>


namespace tksvm {
namespace phase_space {
namespace classifier {

    template <typename Point>
    struct fixed_from_sweep : policy<Point> {
        using typename policy<Point>::point_type;
        using typename policy<Point>::label_type;

        static void define_parameters(alps::params &, std::string const&) {
        }

        fixed_from_sweep(alps::params const& parameters,
                         std::string const&)
        {
            auto sweep_pol = phase_space::sweep::from_parameters<point_type>(
                parameters, "sweep.");
            std::mt19937 rng{parameters["SEED"].as<size_t>()};
            std::generate_n(std::back_inserter(points), sweep_pol->size(),
                [&, p=point_type{}]() mutable {
                    sweep_pol->yield(p, rng);
                    return p;
                });
        }

        virtual label_type operator()(point_type pp) override {
            point::distance<point_type> dist{};
            auto closest_it = std::min_element(points.begin(), points.end(),
                [&](point_type const& lhs, point_type const& rhs) {
                    return dist(lhs, pp) < dist(rhs, pp);
                });
            return {static_cast<double>(closest_it - points.begin())};
        }

        virtual std::string name(label_type const& l) const override {
            if (size_t(l) >= size())
                return policy<point_type>::name(l);
            std::stringstream ss;
            ss << 'P' << (size_t(l) + 1);
            return ss.str();
        }

        virtual size_t size() const override {
            return points.size();
        }
    private:
        std::vector<point_type> points;
    };

}
}
}
