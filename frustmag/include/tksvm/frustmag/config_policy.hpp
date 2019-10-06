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
#include <stdexcept>
#include <string>

#include <alps/params.hpp>

#include <tksvm/config/clustered_policy.hpp>
#include <tksvm/config/policy.hpp>
#include <tksvm/symmetry_policy/none.hpp>
#include <tksvm/symmetry_policy/symmetrized.hpp>

#include <tksvm/frustmag/cluster_policy/lattice.hpp>
#include <tksvm/frustmag/cluster_policy/single.hpp>


namespace tksvm {
namespace frustmag {

template <typename Config, typename Introspector, typename SymmetryPolicy,
          template<typename> typename ClusterPolicy>
using frustmag_config_policy =
    config::clustered_policy<Config,
                             Introspector,
                             SymmetryPolicy,
                             ClusterPolicy<Config>>;

inline void define_config_policy_parameters(alps::params & parameters) {
    parameters
        .define<bool>("symmetrized", true, "use symmetry <S_x S_y> == <S_y S_x>")
        .define<std::string>("cluster", "lattice", "cluster used for SVM config")
        .define<size_t>("rank", "rank of the order parameter tensor");
}


template <typename Config, typename Introspector>
auto config_policy_from_parameters(alps::params const& parameters,
                                   bool unsymmetrize = true)
    -> std::unique_ptr<config::policy<Config, Introspector>>
{
#define CONFPOL_CREATE(CLNAME)                                          \
    return std::unique_ptr<config::policy<Config, Introspector>>(       \
        new frustmag_config_policy<Config, Introspector,                \
            SymmetryPolicy, cluster_policy:: CLNAME >(               \
                rank, element_policy:: CLNAME <Config>{}, unsymmetrize));

#define CONFPOL_BRANCH_SYMM(CLNAME)                                     \
    if (parameters["symmetrized"].as<bool>()) {                         \
        using SymmetryPolicy = symmetry_policy::symmetrized;            \
        CONFPOL_CREATE(CLNAME)                                          \
    } else {                                                            \
        using SymmetryPolicy = symmetry_policy::none;                   \
        CONFPOL_CREATE(CLNAME)                                          \
    }


    // set up SVM configuration policy
    size_t rank = parameters["rank"].as<size_t>();
    std::string clname = parameters["cluster"];
    if (clname == "lattice") {
        CONFPOL_BRANCH_SYMM(lattice)
    } else if (clname == "single") {
        CONFPOL_BRANCH_SYMM(single)
    } else {
        throw std::runtime_error("Invalid cluster spec: " + clname);
    }

#undef CONFPOL_BRANCH_SYMM
#undef CONFPOL_CREATE
}

}
}
