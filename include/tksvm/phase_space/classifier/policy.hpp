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

#include <tksvm/phase_space/label.hpp>


namespace tksvm {
namespace phase_space {
namespace classifier {

    template <typename Point>
    struct policy {
        using point_type = Point;
        using label_type = label::numeric_label<>;

        virtual ~policy() noexcept = default;

        virtual label_type operator()(point_type) = 0;
        virtual std::string name(label_type const& l) const {
            if (size_t(l) == size())
                return "InfT";
            return "None";
        }
        virtual size_t size() const = 0;

        auto get_functor() {
            return [this](point_type pp) {
                return (*this)(pp);
            };
        }

        label_type infinity_label() const {
            return {static_cast<double>(size())};
        }
    };

}
}
}
