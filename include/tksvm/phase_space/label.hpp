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
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>

#include <svm/traits/label_traits.hpp>


namespace tksvm {
namespace phase_space {
namespace label {

    template <size_t nr = svm::DYNAMIC>
    struct numeric_label {
        static const size_t nr_labels = nr;
        static const size_t label_dim = 1;
        numeric_label () : val(0) {}
        template <class Iterator,
                  typename Tag = typename std::iterator_traits<Iterator>::value_type>
        numeric_label(Iterator begin) : val (floor(*begin)) {
            if (val < 0 || val >= nr_labels)
                throw std::runtime_error(static_cast<std::stringstream&>(
                    std::stringstream{} << "invalid label: " << val).str());
        }
        numeric_label(double x) : val (floor(x)) {
            if (val < 0. || val >= nr_labels)
                throw std::runtime_error(static_cast<std::stringstream&>(
                    std::stringstream{} << "invalid label: " << val).str());
        }
        operator double() const { return val; }
        explicit operator size_t() const {
            return static_cast<size_t>(val + 0.5);
        }
        double const * begin() const { return &val; }
        double const * end() const { return &val + 1; }
        friend bool operator== (numeric_label lhs, numeric_label rhs) {
            return lhs.val == rhs.val;
        }
        friend std::ostream & operator<<(std::ostream & os, numeric_label l) {
            return os << l.val;
        }
    private:
        double val;
    };

}
}
}
