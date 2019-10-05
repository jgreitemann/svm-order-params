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

#include <cmath>
#include <limits>


namespace tksvm {
namespace block_reduction {

    constexpr size_t inf = std::numeric_limits<size_t>::infinity();

    template <size_t N>
    struct norm {
        norm & operator+= (double x) {
            sum += std::pow(std::abs(x), N);
            ++M;
            return *this;
        }
        operator double () const {
            return std::pow(sum, 1./N);
        }
    private:
        double sum = 0.;
        size_t M = 0;
    };

    template <>
    struct norm<inf> {
        norm & operator+= (double x) {
            if (std::abs(x) > max)
                max = std::abs(x);
            return *this;
        }
        operator double () const {
            return max;
        }
    private:
        double max = 0.;
    };

    struct sum {
        sum & operator+= (double x) {
            sum += x;
            return *this;
        }
        operator double () const {
            return sum;
        }
    private:
        double sum = 0;
    };
}
}
