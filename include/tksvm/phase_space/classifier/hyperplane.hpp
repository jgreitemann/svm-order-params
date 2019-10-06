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
#include <functional>
#include <numeric>
#include <string>

#include <alps/params.hpp>

#include <tksvm/phase_space/classifier/policy.hpp>


namespace tksvm {
namespace phase_space {
namespace classifier {

    template <typename Point>
    struct hyperplane : policy<Point> {
        using typename policy<Point>::point_type;
        using typename policy<Point>::label_type;

        static void define_parameters(alps::params & params,
                                      std::string const& prefix)
        {
            point_type::define_parameters(params, prefix + "hyperplane.support.");
            point_type::define_parameters(params, prefix + "hyperplane.normal.");
        }

        hyperplane(alps::params const& params,
                   std::string const& prefix)
            : support(params, prefix + "hyperplane.support.")
            , normal(params, prefix + "hyperplane.normal.") {}

        virtual label_type operator()(point_type pp) override {
            std::transform(pp.begin(), pp.end(), support.begin(), pp.begin(),
                           std::minus<>{});
            double res = std::inner_product(pp.begin(), pp.end(),
                                            normal.begin(), 0.);
            return res > 0 ? label_type{1.} : label_type{0.};
        }

        virtual std::string name(label_type const& l) const override {
            if (size_t(l) >= 2)
                return policy<point_type>::name(l);
            return names[size_t(l)];
        }

        virtual size_t size() const override {
            return 2;
        }

    private:
        static const std::string names[];
        point_type support, normal;
    };

    template <typename Point>
    const std::string hyperplane<Point>::names[] = {
        "DISORDERED",
        "ORDERED",
    };

}
}
}
