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
#include <utility>

#include <alps/params.hpp>

#include <tksvm/config/clustered_policy.hpp>
#include <tksvm/config/policy.hpp>
#include <tksvm/symmetry_policy/none.hpp>
#include <tksvm/symmetry_policy/symmetrized.hpp>

#include <tksvm/gauge/cluster_policy/full.hpp>
#include <tksvm/gauge/cluster_policy/single.hpp>
#include <tksvm/gauge/cluster_policy/square.hpp>
#include <tksvm/gauge/element_policy/mono.hpp>
#include <tksvm/gauge/element_policy/triad.hpp>

namespace tksvm {
namespace gauge {

template <typename Config, typename Introspector,
          typename SymmetryPolicy, typename ClusterPolicy>
using gauge_config_policy = config::clustered_policy<Config,
                                                     Introspector,
                                                     SymmetryPolicy,
                                                     ClusterPolicy>;

inline void define_config_policy_parameters(alps::params & parameters) {
    parameters
        .define<std::string>("color", "triad",
                             "use 3 colored spins (triad) or just one (mono)")
        .define<std::string>("cluster", "single", "cluster used for SVM config")
        .define<bool>("symmetrized", true, "use symmetry <l_x m_y> == <m_y l_x>")
        .define<size_t>("rank", "rank of the order parameter tensor");
}


template <typename Config, typename Introspector>
auto config_policy_from_parameters(alps::params const& parameters,
                                   bool unsymmetrize = true)
    -> std::unique_ptr<config::policy<Config, Introspector>>
{
#define CONFPOL_CREATE()                                                \
    return std::unique_ptr<config::policy<Config, Introspector>>(       \
        new gauge_config_policy<                                        \
        Config, Introspector, SymmetryPolicy, ClusterPolicy>(           \
            rank, std::move(elempol), unsymmetrize));                   \


#define CONFPOL_BRANCH_SYMM(CLNAME, CLSIZE)                         \
    using ClusterPolicy =                                           \
        cluster_policy:: CLNAME <BaseElementPolicy, Config>;        \
    using ElementPolicy = typename ClusterPolicy::ElementPolicy;    \
    ElementPolicy elempol{ CLSIZE };                                \
    if (parameters["symmetrized"].as<bool>()) {                     \
        using SymmetryPolicy = symmetry_policy::symmetrized;        \
        CONFPOL_CREATE()                                            \
    } else {                                                        \
        using SymmetryPolicy = symmetry_policy::none;               \
        CONFPOL_CREATE()                                            \
    }                                                               \


#define CONFPOL_BRANCH_CLUSTER() \
    if (clname == "single") {                               \
        CONFPOL_BRANCH_SYMM(single,);                       \
    } else if (clname == "bipartite") {                     \
        CONFPOL_BRANCH_SYMM(square,2);                      \
    } else if (clname == "full") {                          \
        CONFPOL_BRANCH_SYMM(full,(L*L*L));                  \
    } else {                                                \
        throw std::runtime_error("unknown cluster name: "   \
                                 + clname);                 \
    }                                                       \


    // set up SVM configuration policy
    size_t rank = parameters["rank"].as<size_t>();
    size_t L = parameters["length"].as<size_t>();
    std::string clname = parameters["cluster"].as<std::string>();
    std::string elname = parameters["color"].as<std::string>();

    if (elname == "mono") {
        using BaseElementPolicy = element_policy::mono;
        CONFPOL_BRANCH_CLUSTER();
    } else if (elname == "triad") {
        using BaseElementPolicy = element_policy::triad;
        CONFPOL_BRANCH_CLUSTER();
    } else {
        throw std::runtime_error("unknown color setting: " + elname);
    }

#undef CONFPOL_BRANCH_CLUSTER
#undef CONFPOL_BRANCH_SYMM
#undef CONFPOL_CREATE
}

}
}
