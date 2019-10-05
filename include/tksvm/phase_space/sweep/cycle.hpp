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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <alps/params.hpp>
#include <alps/hdf5.hpp>

#include <tksvm/phase_space/sweep/policy.hpp>


namespace tksvm {
namespace phase_space {
namespace sweep {

    template <typename Point>
    struct cycle : public policy<Point> {
        using point_type = typename policy<Point>::point_type;
        using rng_type = typename policy<Point>::rng_type;
        static const size_t MAX_CYCLE = 8;

        static void define_parameters(alps::params & params,
            std::string const& prefix)
        {
            for (size_t i = 1; i <= MAX_CYCLE; ++i)
                point_type::define_parameters(params,
                    format_prefix(i, prefix));
        }

        cycle (std::initializer_list<point_type> il, size_t offset = 0)
            : n(offset)
        {
            for (auto p : il)
                points.push_back(p);
            if (points.empty())
                throw std::runtime_error("cycle sweep policy "
                    "list-initialized but no points supplied");
            n = n % points.size();
        }

        cycle (alps::params const& params,
               std::string const& prefix,
               size_t offset = 0)
            : n(offset)
        {
            for (size_t i = 1; i <= MAX_CYCLE
                && point_type::supplied(params, format_prefix(i, prefix)); ++i)
            {
                points.emplace_back(params, format_prefix(i, prefix));
            }
            if (points.empty())
                throw std::runtime_error("cycle sweep policy initialized "
                    "but no points supplied in parameters");
            n = n % points.size();
        }

        virtual size_t size() const final override {
            return points.size();
        }

        virtual bool yield (point_type & point, rng_type &) final override {
            point = points[n];
            n = (n + 1) % points.size();
            return true;
        }

        virtual void save (alps::hdf5::archive & ar) const final override {
            ar["n"] << n;
        }

        virtual void load (alps::hdf5::archive & ar) final override {
            ar["n"] >> n;
        }

        static std::string format_prefix(size_t i,
            std::string const& prefix)
        {
            std::stringstream ss;
            ss << prefix << "cycle.P" << i << '.';
            return ss.str();
        }

    private:
        std::vector<point_type> points;
        size_t n;
    };

}
}
}
