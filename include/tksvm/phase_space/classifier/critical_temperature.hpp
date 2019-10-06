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

#include <tksvm/phase_space/classifier/policy.hpp>
#include <tksvm/phase_space/point/temperature.hpp>


namespace tksvm {
namespace phase_space {
namespace classifier {

    struct critical_temperature : policy<point::temperature> {
        static void define_parameters(alps::params & params,
                                      std::string const& prefix)
        {
            params.define<double>(prefix + "critical_temperature.temp_crit",
                                  1., "discriminatory temperature");
        }

        critical_temperature(alps::params const& params,
                             std::string const& prefix)
            : temp_crit(params[prefix + "critical_temperature.temp_crit"].as<double>()) {}

        virtual label_type operator() (point_type pp) override {
            return (pp.temp < temp_crit) ? label_type{1.} : label_type{0.};
        }

        virtual std::string name(label_type const& l) const override {
            if (size_t(l) >= 2)
                return classifier::policy<point_type>::name(l);
            return names[size_t(l)];
        }

        virtual size_t size() const override {
            return 2;
        }
    private:
        static constexpr const char * names[] = {
            "DISORDERED",
            "ORDERED",
        };
        double temp_crit;
    };

    constexpr const char * critical_temperature::names[]; // [depr.static_constexpr]

}
}
}

