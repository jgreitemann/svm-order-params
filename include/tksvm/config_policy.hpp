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
#include "indices.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <alps/params.hpp>

#include <boost/multi_array.hpp>


namespace element_policy {

    struct components {
        size_t N;

        constexpr size_t n_block() const { return 1; }
        constexpr size_t range() const { return N * n_block(); }
        constexpr size_t block(size_t) const { return 0; }
        constexpr size_t component(size_t index) const { return index; }
    };

}

namespace cluster_policy {
    template <typename BaseElementPolicy, typename Container>
    struct stride {
        using ElementPolicy = BaseElementPolicy;
        using site_const_iterator = typename Container::const_iterator;

        struct unitcell;
        struct const_iterator {
            const_iterator & operator++ () { sit += range_; return *this; }
            const_iterator operator++ (int) {
                const_iterator old(*this);
                ++(*this);
                return old;
            }
            const_iterator & operator-- () { sit -= range_; return *this; }
            const_iterator operator-- (int) {
                const_iterator old(*this);
                --(*this);
                return old;
            }
            friend bool operator== (const_iterator lhs, const_iterator rhs) { return lhs.sit == rhs.sit; }
            friend bool operator!= (const_iterator lhs, const_iterator rhs) { return lhs.sit != rhs.sit; }
            unitcell operator* () const { return {sit, stride_}; }
            std::unique_ptr<unitcell> operator-> () const {
                return std::unique_ptr<unitcell>(new unitcell(sit, stride_));
            }
            friend stride;
        private:
            const_iterator (site_const_iterator it, size_t range, size_t stride)
                : sit {it}, range_{range}, stride_{stride} {}
            site_const_iterator sit;
            size_t range_, stride_;
        };

        struct unitcell {
            auto operator[](size_t block) const {
                return it + block * stride_;
            }
            friend const_iterator;
        private:
            unitcell (site_const_iterator it, size_t stride)
                : it{it}, stride_{stride} {}
            site_const_iterator it;
            size_t stride_;
        };

        stride(ElementPolicy && elempol, Container const& linear)
            : linear{linear}
            , range_{elempol.range()}
            , stride_{elempol.range() / elempol.n_block()}
        {
        }

        const_iterator begin () const {
            return {linear.begin(), range_, stride_};
        }

        const_iterator end () const {
            return {linear.end(), range_, stride_};
        }

        size_t size () const {
            return std::distance(linear.begin(), linear.end()) / range_;
        }

    private:
        Container const& linear;
        size_t range_, stride_;
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

        bool transform_ind (indices_t &) const {
            return false;
        }

        size_t number_of_equivalents (indices_t const&) const {
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
    virtual size_t n_components () const = 0;
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
                size_t i_out = rearranged_index(i_ind);
                indices_t j_ind(rank_);
                for (size_t j = 0; j < size(); ++j, advance_ind(j_ind)) {
                    do {
                        size_t j_out = rearranged_index(j_ind);
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
                size_t out = rearranged_index(component_indices(ind));
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
                size_t i_out = rearranged_index(i_ind);
                indices_t j_ind(rank_);
                for (size_t j = 0; j < size(); ++j, advance_ind(j_ind)) {
                    do {
                        size_t j_out = rearranged_index(j_ind);
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

    virtual size_t n_components () const override final {
        return ElementPolicy::range() / ElementPolicy::n_block();
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
    using ElementPolicy::block;
    using ElementPolicy::component;

private:
    size_t rearranged_index (indices_t const& ind) const {
        size_t n_block = ElementPolicy::n_block();
        size_t n_component = ElementPolicy::range() / n_block;
        size_t components = 0;
        size_t blocks = 0;
        size_t shift = 1;
        for (auto it = ind.begin(); it != ind.end(); ++it) {
            components *= n_component;
            components += component(*it);
            blocks *= n_block;
            blocks += block(*it);
            shift *= n_component;
        }
        return blocks * shift + components;
    }

    using SymmetryPolicy::transform_ind;

    size_t rank_;
    bool unsymmetrize;
    std::vector<double> weights_;
};


template <typename Config, typename Introspector,
          typename SymmetryPolicy, typename ClusterPolicy>
struct clustered_config_policy
    : public monomial_config_policy<Config, Introspector, SymmetryPolicy,
                                    typename ClusterPolicy::ElementPolicy>
{
    using ElementPolicy = typename ClusterPolicy::ElementPolicy;
    using BasePolicy = monomial_config_policy<Config, Introspector,
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
