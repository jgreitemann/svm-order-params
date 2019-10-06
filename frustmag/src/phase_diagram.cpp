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

#include <cmath>

#include <tksvm/phase_space/classifier/phase_diagram.hpp>
#include <tksvm/phase_space/point/temperature.hpp>

#include <tksvm/frustmag/phase_diagram.hpp>


using namespace tksvm::phase_space;

const typename classifier::phase_diagram<point::temperature>::map_type
classifier::phase_diagram_database<point::temperature>::map {
    {"Kagome", {
        {"Low", {{0.}, {0.00002}}},
        {"Inter", {{0.008}, {0.02}}}
    }},
    {"Kagome2", {
        {"I", {{0.}, {0.004}}},
        {"II", {{0.004}, {0.4}}},
        {"III", {{0.4}, {10.}}},
        {"IV", {{10.}, {10000.}}}
    }},
    {"Kagome3", {
        {"I", {{0.}, {0.004}}},
        {"II", {{0.004}, {0.4}}},
        {"III", {{10.}, {10000.}}}
    }}
};
