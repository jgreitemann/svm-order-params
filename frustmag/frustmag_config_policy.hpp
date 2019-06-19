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

#include "combinatorics.hpp"
#include "config_policy.hpp"
#include "indices.hpp"

#include <array>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <alps/params.hpp>


namespace element_policy {
    template <typename Lattice>
    struct lattice {
        constexpr size_t n_block() const { return Lattice::n_basis; }
        constexpr size_t range() const {
            return Lattice::value_type::size * n_block();
        }
        constexpr size_t block(size_t index) const {
            return index / Lattice::value_type::size;
        }
        constexpr size_t component(size_t index) const {
            return index % Lattice::value_type::size;
        }
    };

    template <typename Lattice>
    struct single {
        constexpr size_t n_block() const { return 1; }
        constexpr size_t range() const {
            return Lattice::value_type::size;
        }
        constexpr size_t block(size_t index) const {
            return 0;
        }
        constexpr size_t component(size_t index) const {
            return index % Lattice::value_type::size;
        }
    };
}

template <typename Config, typename Introspector, typename SymmetryPolicy,
          typename ElementPolicy>
struct frustmag_config_policy
    : public monomial_config_policy<Config, Introspector, SymmetryPolicy,
                                    ElementPolicy>
{
    using BasePolicy = monomial_config_policy<Config, Introspector,
                                              SymmetryPolicy, ElementPolicy>;
    using config_array = typename BasePolicy::config_array;

    using BasePolicy::BasePolicy;

    using BasePolicy::size;
    using BasePolicy::rank;

    virtual std::vector<double> configuration (config_array const& lat) const override final
    {
        std::vector<double> v(size());
        indices_t ind(rank());
        auto w_it = weights().begin();
        for (double & elem : v) {
            elem = 0;
            for (auto const& cell : lat.cells()) {
                for (size_t ic = 0; ic < cell.size() / n_block(); ++ic) {
                    double prod = 1;
                    for (size_t a : ind) {
                        prod *= cell[ic * n_block() + block(a)][component(a)];
                    }
                    elem += prod;
                }
            }
            elem *= *w_it / (lat.cells().size() * config_array::n_basis / n_block());

            advance_ind(ind);
            ++w_it;
        }
        return v;
    }

private:
    using ElementPolicy::block;
    using ElementPolicy::n_block;
    using ElementPolicy::component;
    using BasePolicy::advance_ind;
    using BasePolicy::weights;
};


inline void define_frustmag_config_policy_parameters(alps::params & parameters) {
    parameters
        .define<bool>("symmetrized", true, "use symmetry <S_x S_y> == <S_y S_x>")
        .define<std::string>("cluster", "lattice", "cluster used for SVM config")
        .define<size_t>("rank", "rank of the order parameter tensor");
}


template <typename Config, typename Introspector>
auto frustmag_config_policy_from_parameters(alps::params const& parameters,
                                            bool unsymmetrize = true)
    -> std::unique_ptr<config_policy<Config, Introspector>>
{
#define CONFPOL_CREATE()                                                \
    return std::unique_ptr<config_policy<Config, Introspector>>(        \
        new frustmag_config_policy<Config, Introspector,                \
            SymmetryPolicy, decltype(elempol)>(                         \
                rank, std::move(elempol), unsymmetrize));

#define CONFPOL_BRANCH_SYMM()                                           \
    if (parameters["symmetrized"].as<bool>()) {                         \
        using SymmetryPolicy = symmetry_policy::symmetrized;            \
        CONFPOL_CREATE()                                                \
    } else {                                                            \
        using SymmetryPolicy = symmetry_policy::none;                   \
        CONFPOL_CREATE()                                                \
    }


    // set up SVM configuration policy
    size_t rank = parameters["rank"].as<size_t>();
    std::string clname = parameters["cluster"];
    if (clname == "lattice") {
        element_policy::lattice<Config> elempol{};
        CONFPOL_BRANCH_SYMM()
    } else if (clname == "single") {
        element_policy::single<Config> elempol{};
        CONFPOL_BRANCH_SYMM()
    } else {
        throw std::runtime_error("Invalid cluster spec: " + clname);
    }

#undef CONFPOL_BRANCH_SYMM
#undef CONFPOL_CREATE
}
