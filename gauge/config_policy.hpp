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
#include "indices.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <alps/params.hpp>

#include <boost/multi_array.hpp>

#include <Eigen/Dense>


namespace element_policy {

    struct mono {
        const size_t n_color() const { return 1; }
        const size_t n_block() const { return 1; }
        const size_t range() const { return 3 * n_color(); }
        const size_t sublattice(size_t index) const { return 0; }
        const size_t color(size_t index) const { return 2; }
        const size_t block(size_t index) const { return color(index); }
        const size_t component(size_t index) const { return index; }

        size_t rearranged_index (indices_t const& ind) const {
            size_t components = 0;
            for (auto it = ind.begin(); it != ind.end(); ++it) {
                components *= 3;
                components += component(*it);
            }
            return components;
        }
    };

    struct triad {
        const size_t n_color() const { return 3; }
        const size_t n_block() const { return 3; }
        const size_t range() const { return 3 * n_color(); }
        const size_t sublattice(size_t index) const { return 0; }
        const size_t color(size_t index) const { return index / 3; }
        const size_t block(size_t index) const { return color(index); }
        const size_t component(size_t index) const { return index % 3; }

        size_t rearranged_index (indices_t const& ind) const {
            size_t components = 0;
            size_t colors = 0;
            size_t shift = 1;
            for (auto it = ind.begin(); it != ind.end(); ++it) {
                components *= 3;
                components += component(*it);
                colors *= 3;
                colors += color(*it);
                shift *= 3;
            }
            return colors * shift + components;
        }
    };

    template <typename BaseElementPolicy>
    struct n_partite : private BaseElementPolicy {
        n_partite(size_t n) : n_unit(n) {}

        const size_t n_unitcell() const { return n_unit; }
        const size_t n_color() const { return BaseElementPolicy::n_color(); }
        const size_t n_block() const {
            return n_unitcell() * BaseElementPolicy::n_block();
        }
        const size_t range() const {
            return n_unitcell() * BaseElementPolicy::range();
        }
        const size_t sublattice(size_t index) const {
            return index / BaseElementPolicy::range();
        }
        const size_t color(size_t index) const {
            return BaseElementPolicy::color(index % BaseElementPolicy::range());
        }
        const size_t block(size_t index) const {
            return sublattice(index) * BaseElementPolicy::n_block()
                + BaseElementPolicy::block(index % BaseElementPolicy::range());
        }
        const size_t component(size_t index) const {
            return BaseElementPolicy::component(index % BaseElementPolicy::range());
        }

        size_t rearranged_index (indices_t const& ind) const {
            size_t sublats = 0;
            size_t shift = 1;
            for (auto it = ind.begin(); it != ind.end(); ++it) {
                sublats *= n_unitcell();
                sublats += sublattice(*it);
                shift *= BaseElementPolicy::range();
            }
            indices_t base_ind(ind);
            for (size_t & i : base_ind)
                i = i % BaseElementPolicy::range();
            return sublats * shift + BaseElementPolicy::rearranged_index(base_ind);
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
            typename Container::value_type const& sublattice (size_t i = 0) const {
                if (i != 0)
                    throw std::runtime_error("invalid sublattice index");
                return it[i];
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
            typename Container::value_type const& sublattice (size_t i) const {
                if (i == 0)
                    return root[idx];
                else if (i == 1)
                    return root[idx / L * L + (idx + 1) % L];
                else
                    throw std::runtime_error("invalid sublattice index");
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
            typename Container::value_type const& sublattice (size_t i) const {
                return root[i];
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


namespace symmetry_policy {

    struct none {
        size_t size (size_t range, size_t rank) const {
            return combinatorics::ipow(range, rank);
        }

        void advance_ind (indices_t & ind, size_t range) const {
            auto rit = ind.rbegin();
            ++(*rit);
            while (*rit == range) {
                ++rit;
                if (rit == ind.rend())
                    break;
                ++(*rit);
                auto it = rit.base();
                while (it != ind.end()) {
                    *it = 0;
                    ++it;
                }
            }
        }

        bool transform_ind (indices_t & ind) const {
            return false;
        }

        size_t number_of_equivalents (indices_t const& ind) const {
            return 1;
        }
    };

    struct symmetrized {
        size_t size (size_t range, size_t rank) const {
            return combinatorics::binomial(rank + range - 1, rank);
        }

        void advance_ind (indices_t & ind, size_t range) const {
            auto rit = ind.rbegin();
            ++(*rit);
            while (*rit == range) {
                ++rit;
                if (rit == ind.rend())
                    break;
                ++(*rit);
                auto it = rit.base();
                while (it != ind.end()) {
                    *it = *rit;
                    ++it;
                }
            }
        }

        bool transform_ind (indices_t & ind) const {
            return std::next_permutation(ind.begin(), ind.end());
        }

        size_t number_of_equivalents (indices_t const& ind) const {
            return combinatorics::number_of_permutations(ind);
        }
    };

}

namespace block_reduction {

    constexpr size_t inf = std::numeric_limits<size_t>::infinity();

    template <size_t N>
    struct norm {
        norm & operator+= (double x) {
            sum += std::pow(std::abs(x), N);
            ++M;
            return *this;
        }
        operator double () const {
            return std::pow(sum, 1./N);
        }
    private:
        double sum = 0.;
        size_t M = 0;
    };

    template <>
    struct norm<inf> {
        norm & operator+= (double x) {
            if (std::abs(x) > max)
                max = std::abs(x);
            return *this;
        }
        operator double () const {
            return max;
        }
    private:
        double max = 0.;
    };

    struct sum {
        sum & operator+= (double x) {
            sum += x;
            return *this;
        }
        operator double () const {
            return sum;
        }
    private:
        double sum = 0;
    };
}

template <typename Config, typename Introspector>
struct config_policy {
    using config_array = Config;
    using introspec_t = Introspector;

    using matrix_t = boost::multi_array<double, 2>;

    virtual size_t size () const = 0;
    virtual size_t range () const = 0;
    virtual size_t rank () const = 0;
    virtual std::vector<double> configuration (config_array const&) const = 0;

    virtual matrix_t rearrange (matrix_t const& c) const = 0;
    virtual matrix_t rearrange (introspec_t const& c,
                                indices_t const& bi,
                                indices_t const& bj) const = 0;
    virtual std::pair<matrix_t, matrix_t> block_structure (matrix_t const& c) const = 0;

    virtual indices_t block_indices(indices_t const& ind) const = 0;
    virtual indices_t component_indices(indices_t const& ind) const = 0;

    virtual std::map<indices_t, index_assoc_vec> all_block_indices () const = 0;
};

template <typename Config, typename Introspector,
          typename SymmetryPolicy, typename ElementPolicy>
struct monomial_config_policy
    : public config_policy<Config, Introspector>
    , protected ElementPolicy
    , private SymmetryPolicy
{
    using typename config_policy<Config, Introspector>::matrix_t;
    using typename config_policy<Config, Introspector>::introspec_t;

    monomial_config_policy (size_t rank,
                            ElementPolicy && elempol,
                            bool unsymmetrize = true)
        : ElementPolicy{std::forward<ElementPolicy>(elempol)}
        , rank_(rank)
        , unsymmetrize(unsymmetrize)
        , weights_(size())
    {
        indices_t ind(rank_);
        for (double & w : weights_) {
            w = sqrt(SymmetryPolicy::number_of_equivalents(ind));
            advance_ind(ind);
        }
    }

    virtual size_t size () const override final {
        return SymmetryPolicy::size(ElementPolicy::range(), rank_);
    }

    virtual matrix_t rearrange (matrix_t const& c) const override final {
        symmetry_policy::none no_symm;
        size_t no_symm_size = no_symm.size(ElementPolicy::range(), rank_);
        matrix_t out(boost::extents[no_symm_size][no_symm_size]);
        indices_t i_ind(rank_);
        for (size_t i = 0; i < size(); ++i, advance_ind(i_ind)) {
            do {
                size_t i_out = ElementPolicy::rearranged_index(i_ind);
                indices_t j_ind(rank_);
                for (size_t j = 0; j < size(); ++j, advance_ind(j_ind)) {
                    do {
                        size_t j_out = ElementPolicy::rearranged_index(j_ind);
                        out[i_out][j_out] = c[i][j] / (weights_[i] * weights_[j]);
                    } while (unsymmetrize && transform_ind(j_ind));
                }
            } while (unsymmetrize && transform_ind(i_ind));
        }
        return out;
    }

    virtual matrix_t rearrange (introspec_t const& coeff,
                                indices_t const& bi,
                                indices_t const& bj) const override final {
        symmetry_policy::none no_symm;
        size_t no_symm_size = no_symm.size(ElementPolicy::range()
                                           / ElementPolicy::n_block(), rank_);
        matrix_t out(boost::extents[no_symm_size][no_symm_size]);

        indices_t ind(rank_);
        std::vector<size_t> i_ind_lookup(no_symm_size);
        std::vector<size_t> j_ind_lookup(no_symm_size);
        for (size_t i = 0; i < size(); ++i, advance_ind(ind)) {
            do {
                indices_t ind_block = block_indices(ind);
                size_t out = ElementPolicy::rearranged_index(component_indices(ind));
                if (std::equal(bi.begin(), bi.end(), ind_block.begin()))
                    i_ind_lookup[out] = i;
                if (std::equal(bj.begin(), bj.end(), ind_block.begin()))
                    j_ind_lookup[out] = i;
            } while (unsymmetrize && transform_ind(ind));
        }

#pragma omp parallel for
        for (size_t i_out = 0; i_out < no_symm_size; ++i_out) {
            size_t i = i_ind_lookup[i_out];
            for (size_t j_out = 0; j_out < no_symm_size; ++j_out) {
                size_t j = j_ind_lookup[j_out];
                out[i_out][j_out] = coeff.tensor({i, j})
                    / (weights_[i] * weights_[j]);
            }
        }
        return out;
    }

    std::pair<matrix_t, matrix_t> block_structure (matrix_t const& c) const {
        size_t block_range = combinatorics::ipow(ElementPolicy::n_block(), rank_);
        size_t block_size = combinatorics::ipow(ElementPolicy::range()
                                                / ElementPolicy::n_block(), rank_);
        std::pair<matrix_t, matrix_t> blocks {
            matrix_t (boost::extents[block_range][block_range]),
            matrix_t (boost::extents[block_range][block_range])
        };
        boost::multi_array<block_reduction::norm<2>,2> block_2norms(boost::extents[block_range][block_range]);
        boost::multi_array<block_reduction::sum,2> block_sums(boost::extents[block_range][block_range]);
        indices_t i_ind(rank_);
        for (size_t i = 0; i < size(); ++i, advance_ind(i_ind)) {
            do {
                size_t i_out = ElementPolicy::rearranged_index(i_ind);
                indices_t j_ind(rank_);
                for (size_t j = 0; j < size(); ++j, advance_ind(j_ind)) {
                    do {
                        size_t j_out = ElementPolicy::rearranged_index(j_ind);
                        block_2norms[i_out / block_size][j_out / block_size]
                            += c[i][j] / (weights_[i] * weights_[j]);
                        block_sums[i_out / block_size][j_out / block_size]
                            += c[i][j] / (weights_[i] * weights_[j]);
                    } while (unsymmetrize && transform_ind(j_ind));
                }
            } while (unsymmetrize && transform_ind(i_ind));
        }
        for (size_t i = 0; i < block_range; ++i) {
            for (size_t j = 0; j < block_range; ++j) {
                blocks.first[i][j] = block_2norms[i][j];
                blocks.second[i][j] = block_sums[i][j];
            }
        }
        return blocks;
    }

    virtual size_t range () const override final {
        return ElementPolicy::range();
    }

    virtual size_t rank () const override final {
        return rank_;
    }

    virtual indices_t block_indices(indices_t const& ind) const override final {
        indices_t cind;
        cind.reserve(ind.size());
        std::transform(ind.begin(), ind.end(), std::back_inserter(cind),
                       [this] (size_t a) { return block(a); });
        return cind;
    }

    virtual indices_t component_indices(indices_t const& ind) const override final {
        indices_t cind;
        cind.reserve(ind.size());
        std::transform(ind.begin(), ind.end(), std::back_inserter(cind),
                       [this] (size_t a) { return component(a); });
        return cind;
    }

    virtual std::map<indices_t, index_assoc_vec> all_block_indices () const override final
    {
        std::map<indices_t, index_assoc_vec> b;
        indices_t i_ind(rank());
        for (size_t i = 0; i < size(); ++i, advance_ind(i_ind)) {
            auto it = b.insert({block_indices(i_ind), {}}).first;
            it->second.push_back({i, component_indices(i_ind)});
        }
        return b;
    }

protected:
    void advance_ind (indices_t & ind) const {
        SymmetryPolicy::advance_ind(ind, ElementPolicy::range());
    }

    std::vector<double> const& weights() const {
        return weights_;
    }
    using ElementPolicy::component;

private:
    using ElementPolicy::block;

    using SymmetryPolicy::transform_ind;

    size_t rank_;
    bool unsymmetrize;
    std::vector<double> weights_;
};


struct dummy_introspector {
    double tensor(std::array<size_t, 2>) const {
        throw std::runtime_error("not implemented / don't call");
        return {};
    }
};

template <typename SymmetryPolicy, typename ElementPolicy>
struct block_config_policy
    : public monomial_config_policy<int, dummy_introspector,
                                    SymmetryPolicy, ElementPolicy>
{
    using BasePolicy = monomial_config_policy<int, dummy_introspector,
                                              SymmetryPolicy, ElementPolicy>;
    using BasePolicy::BasePolicy;

    // not implemented
    virtual std::vector<double> configuration (int const&) const override final {
        throw std::runtime_error("not implemented / don't call");
        return {};
    };
};


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

    virtual std::vector<double> configuration (config_array const& R) const override final
    {
        std::vector<double> v(size());
        indices_t ind(rank());
        LatticePolicy lattice(R);
        auto w_it = weights().begin();
        for (double & elem : v) {
            for (auto cell : lattice) {
                double prod = 1;
                for (size_t a : ind)
                    prod *= cell.sublattice(sublattice(a))(color(a), component(a));
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
    using ElementPolicy::sublattice;
    using ElementPolicy::color;
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
