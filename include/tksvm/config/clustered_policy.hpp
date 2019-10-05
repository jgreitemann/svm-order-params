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

#include <vector>

#include <tksvm/config/monomial_policy.hpp>
#include <tksvm/utilities/indices.hpp>


namespace tksvm {
namespace config {

template <typename Config, typename Introspector,
          typename SymmetryPolicy, typename ClusterPolicy>
struct clustered_policy
    : public monomial_policy<Config, Introspector, SymmetryPolicy,
                             typename ClusterPolicy::ElementPolicy>
{
    using ElementPolicy = typename ClusterPolicy::ElementPolicy;
    using BasePolicy = monomial_policy<Config, Introspector,
                                              SymmetryPolicy, ElementPolicy>;
    using config_array = typename BasePolicy::config_array;

    using BasePolicy::BasePolicy;

    using BasePolicy::size;
    using BasePolicy::rank;

    virtual std::vector<double> configuration(config_array const& R) const override
    {
        std::vector<double> v(size());
        indices_t ind(rank());
        ClusterPolicy clusters{ElementPolicy{*this}, R};
        auto w_it = weights().begin();
        for (double & elem : v) {
            for (auto && cell : clusters) {
                double prod = 1;
                for (size_t a : ind)
                    prod *= cell[block(a)][component(a)];
                elem += prod;
            }
            elem *= *w_it / clusters.size();

            advance_ind(ind);
            ++w_it;
        }
        return v;
    }

private:
    using BasePolicy::advance_ind;
    using BasePolicy::weights;
    using ElementPolicy::block;
    using ElementPolicy::component;
};

}
}
