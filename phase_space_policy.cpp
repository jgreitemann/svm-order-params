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

#include "phase_space_policy.hpp"

#include <cmath>

using namespace phase_space;

void classifier::critical_temperature::define_parameters(alps::params & params) {
    params.define<double>("classifier.critical_temperature.temp_crit",
                          1., "decriminatory temperature");
}

classifier::critical_temperature::critical_temperature (alps::params const& params)
    : temp_crit(params["classifier.critical_temperature.temp_crit"].as<double>()) {}

classifier::critical_temperature::label_type classifier::critical_temperature::operator() (point_type pp) {
    return (pp.temp < temp_crit
            ? label::binary::ORDERED
            : label::binary::DISORDERED);
}

typename classifier::D2h_phase_diagram::map_type classifier::D2h_map {
    {"D2h_Ke", {
            {label::D2h::O3, {
                    {1.34, 0.0},
                    {1.32, 0.4},
                    {1.26, 0.8},
                    {1.18, 1.0},
                    {1.14, 1.1},
                    {1.08, 1.2},
                    {1.04, 1.3},
                    {0.78, 1.5},
                    {0.62, 1.6},
                    {0.26, 1.7},
                    {0.0, 1.8},
                    {-0.1, 1.8},
                    {-0.1, -0.1},
                    {1.34, -0.1}
                }},
            {label::D2h::Dinfh, {
                    {1.04, 1.3},
                    {0.78, 1.5},
                    {0.62, 1.6},
                    {0.26, 1.7},
                    {0.0, 1.8},
                    {-0.1, 1.8},
                    {-0.1, 100.0},
                    {0.75, 100.0},
                    {0.77, 2.0},
                    {0.82, 1.8},
                    {0.88, 1.6},
                    {0.96, 1.5}
                }},
            {label::D2h::D2h, {
                    {1.34, -0.1},
                    {1.34, 0.0},
                    {1.32, 0.4},
                    {1.26, 0.8},
                    {1.18, 1.0},
                    {1.14, 1.1},
                    {1.08, 1.2},
                    {1.04, 1.3},
                    {0.96, 1.5},
                    {0.88, 1.6},
                    {0.82, 1.8},
                    {0.77, 2.0},
                    {0.75, 100.0},
                    {100.0, 100.0},
                    {100.0, -0.1}
                }}
        }},
    {"D2h", {
            {label::D2h::O3, {
                    {1.65397566334, 0.0},
                    {1.64426209513, 0.25},
                    {1.60143597214, 0.5},
                    {1.54382850216, 0.75},
                    {1.45961113759, 1.0},
                    {1.36517244262, 1.25},
                    {1.24229797657, 1.5},
                    {1.0, 1.68934826357},
                    {0.75, 1.83607892474},
                    {0.5, 1.91888960978},
                    {0.25, 1.96210011654},
                    {0.0, 1.98317692428},
                    {-0.1, 1.98317692428},
                    {-0.1, -0.1},
                    {1.65397566334, -0.1}
                }},
            {label::D2h::Dinfh, {
                    {1.24229797657, 1.5},
                    {1.0, 1.68934826357},
                    {0.75, 1.83607892474},
                    {0.5, 1.91888960978},
                    {0.25, 1.96210011654},
                    {0.0, 1.98317692428},
                    {-0.1, 1.98317692428},
                    {-0.1, 100.0},
                    {1.06775865351, 100.0},
                    {1.06775865351, 3.0},
                    {1.07246623858, 2.75},
                    {1.07465979274, 2.5},
                    {1.08410022874, 2.25},
                    {1.11361516676, 2.0},
                    {1.14552589345, 1.75}
                }},
            {label::D2h::D2h, {
                    {1.65397566334, -0.1},
                    {1.65397566334, 0.0},
                    {1.64426209513, 0.25},
                    {1.60143597214, 0.5},
                    {1.54382850216, 0.75},
                    {1.45961113759, 1.0},
                    {1.36517244262, 1.25},
                    {1.24229797657, 1.5},
                    {1.14552589345, 1.75},
                    {1.11361516676, 2.0},
                    {1.08410022874, 2.25},
                    {1.07465979274, 2.5},
                    {1.07246623858, 2.75},
                    {1.06775865351, 3.0},
                    {1.06775865351, 100.0},
                    {100.0, 100.0},
                    {100.0, -0.1}
                }}
        }}
};

typename classifier::D3h_phase_diagram::map_type classifier::D3h_map {
    {"D3h", {
            {label::D3h::O3, {
                    {2.245, -0.1},
                    {2.245, 0.0},
                    {2.226, 0.25},
                    {2.164, 0.5},
                    {2.033, 0.75},
                    {1.860, 1.0},
                    {1.630, 1.25},
                    {1.337, 1.5},
                    {0.961, 1.75},
                    {0.5, 1.931},
                    {0.25, 1.982},
                    {0.0, 2.000},
                    {-0.1, 2.000},
                    {-0.1, -0.1},
                }},
            {label::D3h::Dinfh, {
                    {1.841, 100.0},
                    {1.841, 2.5},
                    {1.839, 2.0},
                    {1.863, 1.5},
                    {1.996, 1.0},
                    {2.076, 0.75},
                    {2.164, 0.5},
                    {2.033, 0.75},
                    {1.860, 1.0},
                    {1.630, 1.25},
                    {1.337, 1.5},
                    {0.961, 1.75},
                    {0.5, 1.931},
                    {0.25, 1.982},
                    {0.0, 2.000},
                    {-0.1, 2.000},
                    {-0.1, 100.0},
                }},
            {label::D3h::D3h, {
                    {1.841, 100.0},
                    {1.841, 2.5},
                    {1.839, 2.0},
                    {1.863, 1.5},
                    {1.996, 1.0},
                    {2.076, 0.75},
                    {2.164, 0.5},
                    {2.226, 0.25},
                    {2.245, 0.0},
                    {2.245, -0.1},
                    {100.0, -0.1},
                    {100.0, 100.0},
                }}
        }}
};
