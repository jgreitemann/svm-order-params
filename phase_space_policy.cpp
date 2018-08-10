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

void sweep::gaussian_temperatures::define_parameters(alps::params & params) {
    const std::string prefix = "sweep.gaussian_temperatures.";
    params
        .define<double>(prefix + "temp_step", 1., "maximum change of temperature")
        .define<double>(prefix + "temp_center", 1., "center temperature")
        .define<double>(prefix + "temp_sigma", 1., "std. deviation of temperature")
        .define<double>(prefix + "temp_min", 0.0, "minimum value of temperature")
        .define<double>(prefix + "temp_max", std::numeric_limits<double>::max(),
                        "maximum value of temperature");
}

sweep::gaussian_temperatures::gaussian_temperatures (alps::params const& params)
    : temp_center(params["sweep.gaussian_temperatures.temp_center"].as<double>())
    , temp_min(params["sweep.gaussian_temperatures.temp_min"].as<double>())
    , temp_max(params["sweep.gaussian_temperatures.temp_max"].as<double>())
    , temp_step(params["sweep.gaussian_temperatures.temp_step"].as<double>())
    , temp_sigma_sq(pow(params["sweep.gaussian_temperatures.temp_sigma"].as<double>(), 2))
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

void sweep::uniform_temperatures::define_parameters(alps::params & params) {
    const std::string prefix = "sweep.uniform_temperatures.";
    params
        .define<double>(prefix + "temp_step", 1., "maximum change of temperature")
        .define<double>(prefix + "temp_min", 0.0, "minimum value of temperature")
        .define<double>(prefix + "temp_max", std::numeric_limits<double>::max(),
                        "maximum value of temperature");
}

sweep::uniform_temperatures::uniform_temperatures (alps::params const& params)
    : temp_min(params["sweep.uniform_temperatures.temp_min"].as<double>())
    , temp_max(params["sweep.uniform_temperatures.temp_max"].as<double>())
    , temp_step(params["sweep.uniform_temperatures.temp_step"].as<double>())
{}

bool sweep::uniform_temperatures::yield (point_type & point, rng_type & rng) {
    using uniform_t = std::uniform_real_distribution<double>;

    double delta_temp;
    do {
        delta_temp = uniform_t {-temp_step, temp_step} (rng);
    } while (point.temp + delta_temp < temp_min || point.temp + delta_temp > temp_max);

    point.temp += delta_temp;
    return true;
}

void sweep::equidistant_temperatures::define_parameters(alps::params & params) {
    const std::string prefix = "sweep.equidistant_temperatures.";
    params
        .define<double>(prefix + "temp_min", 0.0, "minimum value of temperature")
        .define<double>(prefix + "temp_max", std::numeric_limits<double>::max(),
                        "maximum value of temperature");
}

sweep::equidistant_temperatures::equidistant_temperatures (alps::params const& params,
                                                           size_t N, size_t offset)
    : N(N)
    , n((offset % N == 0) ? 0 : (offset % N - 1))
    , temp_max(params["sweep.equidistant_temperatures.temp_max"].as<double>())
    , temp_step((params["sweep.equidistant_temperatures.temp_max"].as<double>()
                 - params["sweep.equidistant_temperatures.temp_min"].as<double>()) / (N-1))
    , cooling(offset % N != 0)
{}

bool sweep::equidistant_temperatures::yield (point_type & point, rng_type &) {
    if ((n == 0 && !cooling) || (n == N-1 && cooling)) {
        cooling = !cooling;
        point.temp = temp_max - n * temp_step;
        return false;
    } else {
        n += cooling ? 1 : -1;
        point.temp = temp_max - n * temp_step;
        return true;
    }
}
