// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2018  Jonas Greitemann, Ke Liu, and Lode Pollet

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

void sweep::define_parameters (alps::params & params) {
    params
        .define<double>("temp_step", 0.25, "maximum change of temperature")
        .define<double>("temp_crit", 2.269185, "critical temperature")
        .define<double>("temp_sigma", 1.0, "std. deviation of temperature")
        .define<double>("temp_min", 0.0, "minimum value of temperature")
        .define<double>("temp_max", std::numeric_limits<double>::max(),
                        "maximum value of temperature");
}

classifier::critical_temperature::critical_temperature (alps::params const& params)
    : temp_crit(params["temp_crit"].as<double>()) {}

classifier::critical_temperature::label_type classifier::critical_temperature::operator() (point_type pp) {
    return (pp.temp < temp_crit
            ? label::binary::ORDERED
            : label::binary::DISORDERED);
}

sweep::gaussian_temperatures::gaussian_temperatures (alps::params const& params)
    : temp_center(params["temp_crit"].as<double>())
    , temp_min(params["temp_min"].as<double>())
    , temp_max(params["temp_max"].as<double>())
    , temp_step(params["temp_step"].as<double>())
    , temp_sigma_sq(pow(params["temp_sigma"].as<double>(), 2))
{}

bool sweep::gaussian_temperatures::yield (point_type & point, rng_type & rng) {
    using uniform_t = std::uniform_real_distribution<double>;
    double delta_temp;
    do {
        delta_temp = uniform_t {-temp_step, temp_step} (rng);
    } while (point.temp + delta_temp < temp_min || point.temp + delta_temp > temp_max);
    double ratio = exp(-(2. * (point.temp-temp_center) + delta_temp) * delta_temp
                       / 2. / temp_sigma_sq);
    if (ratio > 1 || uniform_t {} (rng) < ratio) {
        point.temp += delta_temp;
        return true;
    }
    return false;
}

