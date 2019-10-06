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

// #include "combinatorics.hpp"
// #include "indices.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

#include <boost/multi_array.hpp>

#include <combinatorics/ipow.hpp>

#include <tksvm/block_reduction.hpp>
#include <tksvm/config/policy.hpp>
#include <tksvm/symmetry_policy/none.hpp>


namespace tksvm {
namespace config {

template <typename Config, typename Introspector,
          typename SymmetryPolicy, typename ElementPolicy>
struct monomial_policy
    : public policy<Config, Introspector>
    , protected ElementPolicy
    , private SymmetryPolicy
{
    using typename policy<Config, Introspector>::matrix_t;
    using typename policy<Config, Introspector>::introspec_t;

    monomial_policy (size_t rank,
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

}
}
