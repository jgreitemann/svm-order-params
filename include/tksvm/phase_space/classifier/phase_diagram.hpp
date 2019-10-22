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

#include <initializer_list>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <alps/params.hpp>

#include <tksvm/phase_space/classifier/policy.hpp>
#include <tksvm/phase_space/point/temperature.hpp>
#include <tksvm/utilities/polygon.hpp>


namespace tksvm {
namespace phase_space {
namespace classifier {

    template <typename Point>
    struct phase_diagram;

    template <typename Point>
    struct phase_diagram_database {
        static const typename phase_diagram<Point>::map_type map;
    };

    template <typename Point>
    const typename phase_diagram<Point>::map_type
    phase_diagram_database<Point>::map {};

    template <typename Point>
    struct phase_diagram : policy<Point> {
        using typename policy<Point>::point_type;
        using typename policy<Point>::label_type;
        using database_type = phase_diagram_database<point_type>;
        using map_type = std::map<std::string, phase_diagram>;
        using pair_type = std::pair<std::string, polygon<point_type>>;

        static void define_parameters(alps::params & params,
                                      std::string const& prefix)
        {
            params.define<std::string>(prefix + "phase_diagram.name", "",
                                       "key of the phase diagram map entry");
        }

        phase_diagram(alps::params const& params,
                      std::string const& prefix)
            : phase_diagram([&] {
                    try {
                        return database_type::map.at(
                            params[prefix + "phase_diagram.name"]);
                    } catch (...) {
                        std::stringstream ss;
                        ss << "unknown phase diagram \""
                           << params[prefix + "phase_diagram.name"].as<std::string>()
                           << "\"";
                        throw std::runtime_error(ss.str());
                    }
                }()) {}

        phase_diagram(std::initializer_list<pair_type> il) {
            pairs.reserve(il.size());
            for (auto const& p : il)
                pairs.push_back(p);
        }

        virtual label_type operator()(point_type pp) override {
            double l = 0.;
            for (auto const& p : pairs) {
                if (p.second.is_inside(pp))
                    return {l};
                l += 1;
            }
            return label_type{static_cast<double>(pairs.size() + 1)};
        }

        virtual std::string name(label_type const& l) const override {
            if (size_t(l) >= size())
                return policy<point_type>::name(l);
            return pairs[size_t(l)].first;
        }

        virtual size_t size() const override {
            return pairs.size();
        }
    private:
        std::vector<pair_type> pairs;
    };

}
}
}
