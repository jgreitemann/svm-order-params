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

#include <sstream>
#include <string>

#include <alps/params.hpp>

#include <tksvm/phase_space/classifier/policy.hpp>


namespace tksvm {
namespace phase_space {
namespace classifier {

    template <typename Point>
    struct orthants : policy<Point> {
        using typename policy<Point>::point_type;
        using typename policy<Point>::label_type;

        static void define_parameters(alps::params & params,
                                      std::string const& prefix)
        {
            point_type::define_parameters(params, prefix + "orthants.");
        }

        orthants(alps::params const& params,
                 std::string const& prefix)
            : origin(params, prefix + "orthants.") {}
        virtual label_type operator()(point_type pp) override {
            size_t res = 0;
            auto oit = origin.begin();
            for (double x : pp) {
                res *= 2;
                if (x > *oit)
                    res += 1;
                ++oit;
            }
            return label_type {double(res)};
        }
        virtual std::string name(label_type const& l) const override {
            if (size_t(l) >= size())
                return policy<point_type>::name(l);
            std::stringstream ss;
            ss << l;
            return ss.str();
        }
        virtual size_t size() const override {
            return 1 << point_type::label_dim;
        }
    private:
        point_type origin;
    };

}
}
}
