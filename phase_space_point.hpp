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
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>

#include <alps/params.hpp>

namespace phase_space {
    namespace point {

        struct temperature {
            static const size_t label_dim = 1;
            using iterator = double *;
            using const_iterator = double const *;

            static void define_parameters(alps::params & params, std::string prefix="") {
                params.define<double>(prefix + "temp", 1., "temperature");
            }

            static bool supplied(alps::params const& params, std::string prefix="") {
                return params.supplied(prefix + "temp");
            }

            temperature() : temp(-1) {}
            temperature(double temp) : temp(temp) {}
            temperature(alps::params const& params, std::string prefix="")
                : temp(params[prefix + "temp"].as<double>()) {}

            template <class Iterator>
            temperature(Iterator begin) : temp(*begin) {}

            const_iterator begin() const { return &temp; }
            iterator begin() { return &temp; }
            const_iterator end() const { return &temp + 1; }
            iterator end() { return &temp + 1; }

            double temp;
        };

        struct J1J3 {
            static const size_t label_dim = 2;
            using iterator = double *;
            using const_iterator = double const *;

            static void define_parameters(alps::params & params, std::string prefix="") {
                params
                    .define<double>(prefix + "J1", 0., "J1 coupling")
                    .define<double>(prefix + "J3", 0., "J3 coupling");
            }

            static bool supplied(alps::params const& params, std::string prefix="") {
                return params.supplied(prefix + "J1")
                    && params.supplied(prefix + "J3");
            }

            J1J3(alps::params const& params, std::string prefix="")
                : J{params[prefix + "J1"].as<double>(),
                    params[prefix + "J3"].as<double>()} {}

            J1J3() : J{-1, -1} {}
            J1J3(double J1, double J3) : J{J1, J3} {}

            template <class Iterator>
            J1J3(Iterator begin) : J {*begin, *(++begin)} {}

            const_iterator begin() const { return J; }
            iterator begin() { return J; }
            const_iterator end() const { return J + 2; }
            iterator end() { return J + 2; }

            double const& J1() const { return J[0]; }
            double & J1() { return J[0]; }
            double const& J3() const { return J[1]; }
            double & J3() { return J[1]; }

            double J[2];
        };

        template <typename Point>
        bool operator== (Point const& lhs, Point const& rhs) {
            return std::equal(lhs.begin(), lhs.end(), rhs.begin());
        }

        template <typename Point>
        bool operator!= (Point const& lhs, Point const& rhs) {
            return !(lhs == rhs);
        }

        template <typename Point>
        bool operator< (Point const& lhs, Point const& rhs) {
            return std::lexicographical_compare(
                lhs.begin(), lhs.end(),
                rhs.begin(), rhs.end());
        }

        template <typename Point,
                  typename = std::enable_if_t<(Point::label_dim > 0)>>
        std::ostream& operator<< (std::ostream & os, Point const& p) {
            auto it = p.begin();
            os << '(' << *(it++);
            for (; it != p.end(); ++it)
                os << ", " << *it;
            return os << ')';
        }

        template <typename Point>
        struct distance {
            double operator() (Point const& lhs, Point const& rhs) const {
                return sqrt(std::inner_product(
                    lhs.begin(), lhs.end(),
                    rhs.begin(), 0.,
                    std::plus<>{},
                    [](double a, double b) {
                        return (a - b) * (a - b);
                    }));
            }
        };

        template <typename Point>
        struct infinity {
            Point operator()() const {
                Point p{};
                std::fill(p.begin(), p.end(), 999999.);
                return p;
            }
        };
    }
}
