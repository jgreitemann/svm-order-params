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
#include <limits>
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
            point::distance<point_type> dist;
            auto process = [&](alps::params const& parameters) {
                auto sweep_pol = phase_space::sweep::from_parameters<point_type>(
                    parameters, "sweep.");
                std::mt19937 rng{parameters["SEED"].as<std::mt19937::result_type>()};
                point_type p;
                for (size_t i = 0; i < sweep_pol->size(); ++i) {
                    sweep_pol->yield(p, rng);
                    double min_dist = std::numeric_limits<double>::max();
                    for (auto const& pp : points)
                        min_dist = std::min(min_dist, dist(p, pp));
                    if (min_dist > 1e-5)
                        points.push_back(p);
                }
            };
            process(parameters);
            std::stringstream merge_is{parameters["merge"].as<std::string>()};
            for (std::string name; std::getline(merge_is, name, ':');) {
                const char* argv[] = {"", name.c_str()};
                alps::params merged_params(2, argv);
                process(merged_params);
            }
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
