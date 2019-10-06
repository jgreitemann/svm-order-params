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

#include <array>
#include <stdexcept>
#include <vector>

#include <tksvm/config/monomial_policy.hpp>


namespace tksvm {
namespace config {

struct dummy_introspector {
    double tensor(std::array<size_t, 2>) const {
        throw std::runtime_error("not implemented / don't call");
        return {};
    }
};

template <typename SymmetryPolicy, typename ElementPolicy>
struct block_policy
    : public monomial_policy<int, dummy_introspector,
                             SymmetryPolicy, ElementPolicy>
{
    using BasePolicy = monomial_policy<int, dummy_introspector,
                                              SymmetryPolicy, ElementPolicy>;
    using BasePolicy::BasePolicy;

    // not implemented
    virtual std::vector<double> configuration (int const&) const override final {
        throw std::runtime_error("not implemented / don't call");
        return {};
    };
};

}
}