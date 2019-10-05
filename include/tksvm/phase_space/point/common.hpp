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
#include <type_traits>


namespace tksvm {
namespace phase_space {
namespace point {

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

}
}
}
