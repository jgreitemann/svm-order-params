// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2018  Jonas Greitemann, Ke Liu, and Lode Pollet

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
#include <stdexcept>
#include <string>
#include <vector>

#include <alps/params.hpp>


namespace element_policy {
    struct Z2_site {
        Z2_site(size_t L) : L(L) {}
        const size_t n_block() const { return L; }
        const size_t range() const { return L; }
        const size_t block(size_t index) const { return index; }
        const size_t component(size_t index) const { return 0; }

        size_t rearranged_index (indices_t const& ind) const {
            return std::accumulate(ind.begin(), ind.end(), 0,
                                   [&] (size_t a, size_t b) { return L*a + b; });
        }

    private:
        size_t L;
    };
}

template <typename Config, typename Introspector, typename SymmetryPolicy>
struct ising_config_policy
    : public monomial_config_policy<Config, Introspector, SymmetryPolicy,
                                    element_policy::Z2_site>
{
    using ElementPolicy = element_policy::Z2_site;
    using BasePolicy = monomial_config_policy<Config, Introspector,
                                              SymmetryPolicy, ElementPolicy>;
    using config_array = typename BasePolicy::config_array;

    using BasePolicy::BasePolicy;

    using BasePolicy::size;
    using BasePolicy::rank;

    virtual std::vector<double> configuration (config_array const& spins) const override final
    {
        std::vector<double> v(size());
        indices_t ind(rank());
        auto w_it = weights().begin();
        for (double & elem : v) {
            elem = 1;
            for (size_t a : ind)
                elem *= spins.data()[a];
            elem *= *w_it;

            advance_ind(ind);
            ++w_it;
        }
        return v;
    }

private:
    using BasePolicy::advance_ind;
    using BasePolicy::weights;
};


inline void define_ising_config_policy_parameters(alps::params & parameters) {
    parameters
        .define<bool>("symmetrized", true, "use symmetry <S_x S_y> == <S_y S_x>")
        .define<size_t>("rank", "rank of the order parameter tensor");
}


template <typename Config, typename Introspector>
auto ising_config_policy_from_parameters(alps::params const& parameters,
                                         bool unsymmetrize = true)
    -> std::unique_ptr<config_policy<Config, Introspector>>
{
#define CONFPOL_CREATE()                                                \
    return std::unique_ptr<config_policy<Config, Introspector>>(        \
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
