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
#include <stdexcept>
#include <string>
#include <vector>

#include <alps/params.hpp>


namespace element_policy {

    struct mono : components {
        mono() : components{3} {}
        static constexpr size_t n_color() { return 1; }
        static constexpr size_t sublattice_of_block(size_t) { return 0; }
        static constexpr size_t color_of_block(size_t) { return 2; }
    };

    struct triad {
        constexpr size_t n_block() const { return 3; }
        constexpr size_t range() const { return 9; }
        constexpr size_t block(size_t index) const { return index / 3; }
        constexpr size_t component(size_t index) const { return index % 3; }
        static constexpr size_t n_color() { return 3; }
        static constexpr size_t sublattice_of_block(size_t) { return 0; }
        static constexpr size_t color_of_block(size_t block) { return block; }
    };

    template <typename BaseElementPolicy>
    struct n_partite : private BaseElementPolicy {
        n_partite(size_t n) : n_unit(n) {}

        constexpr size_t n_block() const {
            return n_unit * BaseElementPolicy::n_block();
        }
        constexpr size_t range() const {
            return n_unit * BaseElementPolicy::range();
        }
        constexpr size_t block(size_t index) const {
            return (index / BaseElementPolicy::range()) * BaseElementPolicy::n_block()
                + BaseElementPolicy::block(index % BaseElementPolicy::range());
        }
        constexpr size_t component(size_t index) const {
            return BaseElementPolicy::component(index % BaseElementPolicy::range());
        }
        static constexpr size_t sublattice_of_block(size_t block) {
            return block / BaseElementPolicy::n_color();
        }
        static constexpr size_t color_of_block(size_t block) {
            return BaseElementPolicy::color_of_block(block % BaseElementPolicy::n_color());
        }
    private:
        const size_t n_unit;
    };
}

namespace lattice {

    template <typename BaseElementPolicy, typename Container>
    struct single {
        using ElementPolicy = BaseElementPolicy;
        using site_const_iterator = typename Container::const_iterator;

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
            friend single;
        private:
            const_iterator (site_const_iterator it) : sit {it} {}
            site_const_iterator sit;
        };

        struct unitcell {
            auto operator[](size_t block) const {
                typename Container::const_reference mat = *it;
                return mat.col(ElementPolicy::color_of_block(block));
            }
            friend const_iterator;
        private:
            unitcell (site_const_iterator it) : it {it} {}
            site_const_iterator it;
        };

        single (Container const& linear) : linear(linear) {}

        const_iterator begin () const {
            return {linear.begin()};
        }

        const_iterator end () const {
            return {linear.end()};
        }

        size_t size () const {
            return linear.size();
        }

    private:
        Container const& linear;
    };

    template <typename BaseElementPolicy, typename Container, size_t DIM = 3>
    struct square {
        using ElementPolicy = element_policy::n_partite<BaseElementPolicy>;
        using site_const_iterator = typename Container::const_iterator;
        using coord_t = std::array<size_t, DIM>;

        struct unitcell;
        struct const_iterator {
            const_iterator & operator++ () {
                coord[0] += 2;
                if (coord[0] >= L) {
                    size_t i;
                    for (i = 1; i < coord.size(); ++i) {
                        ++coord[i];
                        if (coord[i] < L)
                            break;
                        coord[i] = 0;
                    }
                    if (i == coord.size()) {
                        coord[0] = L;
                    } else {
                        size_t sum = 0;
                        for (i = 1; i < coord.size(); ++i) {
                            sum += coord[i];
                        }
                        coord[0] = sum % 2;
                    }
                }
                return *this;
            }
            const_iterator operator++ (int) {
                const_iterator old(*this);
                ++(*this);
                return old;
            }
            friend bool operator== (const_iterator lhs, const_iterator rhs) {
                return (lhs.coord == rhs.coord
                        && lhs.root == rhs.root
                        && lhs.L == rhs.L);
            }
            friend bool operator!= (const_iterator lhs, const_iterator rhs) { return !(lhs == rhs); }
            unitcell operator* () const { return {root, lin_index(), L}; }
            std::unique_ptr<unitcell> operator-> () const {
                return std::make_unique<unitcell>(root, lin_index(), L);
            }
            friend square;
        private:
            const_iterator (site_const_iterator it, coord_t c, size_t L)
                : root{it}, coord{c}, L{L} {}
            size_t lin_index () const {
                size_t sum = 0;
                for (size_t c : coord) {
                    sum *= L;
                    sum += c;
                }
                return sum;
            }
            site_const_iterator root;
            coord_t coord;
            size_t L;
        };

        struct unitcell {
            auto operator[](size_t block) const {
                size_t subl = ElementPolicy::sublattice_of_block(block);
                size_t color = ElementPolicy::color_of_block(block);
                typename Container::const_reference mat =
                    root[subl ? (idx / L * L + (idx + 1) % L) : idx];
                return mat.col(color);
            }
            friend const_iterator;
        private:
            unitcell (site_const_iterator it, size_t idx, size_t L)
                : root{it}, idx{idx}, L{L} {}
            site_const_iterator root;
            size_t idx;
            size_t L;
        };

        square (Container const& linear) : linear(linear) {
            L = static_cast<size_t>(pow(linear.size() + 0.5, 1./DIM));
            if (combinatorics::ipow(L, DIM) != linear.size())
                throw std::runtime_error("linear configuration size doesn't match DIM");
            if (L % 2 != 0)
                throw std::runtime_error("lattice not bipartite w.r.t. PBCs");
        }

        const_iterator begin () const {
            return {linear.begin(), {0}, L};
        }

        const_iterator end () const {
            return {linear.begin(), {L}, L};
        }

        size_t size () const {
            return linear.size() / 2;
        }

    private:
        Container const& linear;
        size_t L;
    };


    template <typename BaseElementPolicy, typename Container>
    struct full {
        using ElementPolicy = element_policy::n_partite<BaseElementPolicy>;
        using site_const_iterator = typename Container::const_iterator;

        struct unitcell;
        struct const_iterator {
            const_iterator & operator++ () {
                root += size;
                return *this;
            }
            const_iterator operator++ (int) {
                const_iterator old(*this);
                ++(*this);
                return old;
            }
            friend bool operator== (const_iterator lhs, const_iterator rhs) {
                return lhs.root == rhs.root;
            }
            friend bool operator!= (const_iterator lhs, const_iterator rhs) { return !(lhs == rhs); }
            unitcell operator* () const { return {root}; }
            std::unique_ptr<unitcell> operator-> () const {
                return std::make_unique<unitcell>(root);
            }
            site_const_iterator root;
            size_t size;
        };

        struct unitcell {
            auto operator[](size_t block) const {
                size_t subl = ElementPolicy::sublattice_of_block(block);
                size_t color = ElementPolicy::color_of_block(block);
                typename Container::const_reference mat = root[subl];
                return mat.col(color);
            }
            site_const_iterator root;
        };

        full (Container const& linear) : linear(linear) {}

        const_iterator begin () const {
            return {linear.begin(), linear.size()};
        }

        const_iterator end () const {
            return {linear.end(), linear.size()};
        }

        size_t size () const {
            return 1;
        }

    private:
        Container const& linear;
    };

}


template <typename Config, typename Introspector,
          typename SymmetryPolicy, typename LatticePolicy>
struct gauge_config_policy
    : public monomial_config_policy<Config, Introspector, SymmetryPolicy,
                                    typename LatticePolicy::ElementPolicy>
{
    using ElementPolicy = typename LatticePolicy::ElementPolicy;
    using BasePolicy = monomial_config_policy<Config, Introspector,
                                              SymmetryPolicy, ElementPolicy>;
    using config_array = typename BasePolicy::config_array;

    using BasePolicy::BasePolicy;

    using BasePolicy::size;
    using BasePolicy::rank;

    virtual std::vector<double> configuration(config_array const& R) const override final
    {
        std::vector<double> v(size());
        indices_t ind(rank());
        LatticePolicy lattice{R};
        auto w_it = weights().begin();
        for (double & elem : v) {
            for (auto && cell : lattice) {
                double prod = 1;
                for (size_t a : ind)
                    prod *= cell[block(a)][component(a)];
                elem += prod;
            }
            elem *= *w_it / lattice.size();

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


inline void define_gauge_config_policy_parameters(alps::params & parameters) {
    parameters
        .define<std::string>("color", "triad",
                             "use 3 colored spins (triad) or just one (mono)")
        .define<std::string>("cluster", "single", "cluster used for SVM config")
        .define<bool>("symmetrized", true, "use symmetry <l_x m_y> == <m_y l_x>")
        .define<size_t>("rank", "rank of the order parameter tensor");
}


template <typename Config, typename Introspector>
auto gauge_config_policy_from_parameters(alps::params const& parameters,
                                         bool unsymmetrize = true)
    -> std::unique_ptr<config_policy<Config, Introspector>>
{
#define CONFPOL_CREATE()                                                \
    return std::unique_ptr<config_policy<Config, Introspector>>(        \
        new gauge_config_policy<                                        \
        Config, Introspector, SymmetryPolicy, LatticePolicy>(           \
            rank, std::move(elempol), unsymmetrize));                   \


#define CONFPOL_BRANCH_SYMM(LATNAME, CLSIZE)                        \
    using LatticePolicy = lattice:: LATNAME <BaseElementPolicy,     \
                                             Config>;               \
    using ElementPolicy = typename LatticePolicy::ElementPolicy;    \
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
