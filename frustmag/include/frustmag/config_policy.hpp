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

namespace cluster_policy {
    template <typename Lattice>
    struct frustmag_single {
        using ElementPolicy = typename element_policy::single<Lattice>;
        using site_const_iterator = typename Lattice::const_iterator;

        struct unitcell;
        struct const_iterator {
            const_iterator & operator++ () { ++sit; return *this; }
            const_iterator operator++ (int) {
                const_iterator old(*this);
                ++(*this);
                return old;
            }
            const_iterator & operator-- () { --sit; return *this; }
            const_iterator operator-- (int) {
                const_iterator old(*this);
                --(*this);
                return old;
            }
            friend bool operator== (const_iterator lhs, const_iterator rhs) { return lhs.sit == rhs.sit; }
            friend bool operator!= (const_iterator lhs, const_iterator rhs) { return lhs.sit != rhs.sit; }
            unitcell operator* () const { return {sit}; }
            std::unique_ptr<unitcell> operator-> () const {
                return std::unique_ptr<unitcell>(new unitcell(sit));
            }
            friend frustmag_single;
        private:
            const_iterator (site_const_iterator it) : sit {it} {}
            site_const_iterator sit;
        };

        struct unitcell {
            auto operator[](size_t) const {
                return *it;
            }
            friend const_iterator;
        private:
            unitcell (site_const_iterator it) : it{it} {}
            site_const_iterator it;
        };

        frustmag_single(ElementPolicy, Lattice const& sites) : sites{sites} {}

        const_iterator begin () const {
            return {sites.begin()};
        }

        const_iterator end () const {
            return {sites.end()};
        }

        size_t size () const {
            return sites.size();
        }

    private:
        Lattice const& sites;
        size_t range_, stride_;
    };

    template <typename Lattice>
    struct frustmag_lattice {
        using ElementPolicy = element_policy::lattice<Lattice>;
        using const_iterator = typename Lattice::unitcell_collection_type::const_iterator;

        frustmag_lattice(ElementPolicy, Lattice const& lat)
            : cells_{lat.cells()}
        {
        }

        const_iterator begin() const {
            return cells_.begin();
        }

        const_iterator end() const {
            return cells_.end();
        }

        auto size() const {
            return cells_.size();
        }

    private:
        typename Lattice::unitcell_collection_type const& cells_;
    };
}

template <typename Config, typename Introspector, typename SymmetryPolicy,
          template<typename> typename ClusterPolicy>
using frustmag_config_policy =
    clustered_config_policy<Config,
                            Introspector,
                            SymmetryPolicy,
                            ClusterPolicy<Config>>;

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
#define CONFPOL_CREATE(CLNAME)                                          \
    return std::unique_ptr<config_policy<Config, Introspector>>(        \
        new frustmag_config_policy<Config, Introspector,                \
            SymmetryPolicy, cluster_policy::frustmag_##CLNAME >(        \
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
