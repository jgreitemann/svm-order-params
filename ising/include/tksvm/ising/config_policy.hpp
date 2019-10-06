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

#include <memory>
#include <utility>

#include <alps/params.hpp>

#include <tksvm/cluster_policy/stride.hpp>
#include <tksvm/config/clustered_policy.hpp>
#include <tksvm/symmetry_policy/none.hpp>
#include <tksvm/symmetry_policy/symmetrized.hpp>

#include <tksvm/ising/element_policy/Z2_site.hpp>


namespace tksvm {
namespace ising {

template <typename Config, typename Introspector, typename SymmetryPolicy>
using ising_config_policy =
    config::clustered_policy<Config,
                             Introspector,
                             SymmetryPolicy,
                             cluster_policy::stride<element_policy::Z2_site, Config>>;


inline void define_config_policy_parameters(alps::params & parameters) {
    parameters
        .define<bool>("symmetrized", true, "use symmetry <S_x S_y> == <S_y S_x>")
        .define<size_t>("rank", "rank of the order parameter tensor");
}


template <typename Config, typename Introspector>
auto config_policy_from_parameters(alps::params const& parameters,
                                   bool unsymmetrize = true)
    -> std::unique_ptr<config::policy<Config, Introspector>>
{
#define CONFPOL_CREATE()                                                \
    return std::unique_ptr<config::policy<Config, Introspector>>(        \
        new ising_config_policy<Config, Introspector, SymmetryPolicy>(  \
            rank, std::move(elempol), unsymmetrize));                   \


    // set up SVM configuration policy
    size_t rank = parameters["rank"].as<size_t>();
    size_t L = parameters["length"].as<size_t>();

    element_policy::Z2_site elempol{ L * L };
    if (parameters["symmetrized"].as<bool>()) {
        using SymmetryPolicy = symmetry_policy::symmetrized;
        CONFPOL_CREATE()
    } else {
        using SymmetryPolicy = symmetry_policy::none;
        CONFPOL_CREATE()
    }


#undef CONFPOL_CREATE
}

}
}
