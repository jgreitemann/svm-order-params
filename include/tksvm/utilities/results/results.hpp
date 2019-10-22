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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include <boost/multi_array.hpp>

#include <tksvm/symmetry_policy/none.hpp>
#include <tksvm/utilities/indices.hpp>


namespace tksvm {
namespace results {

    struct index_rule {
        using rule_ptr = std::unique_ptr<index_rule>;
        virtual ~index_rule() noexcept = default;
        virtual bool operator() (indices_t const&, indices_t const&) const = 0;
        virtual rule_ptr clone () const = 0;
    };

    using rule_ptr = index_rule::rule_ptr;

    struct delta_rule : public index_rule {
        virtual bool operator() (indices_t const& i_ind, indices_t const& j_ind) const override;
        virtual rule_ptr clone () const override;
        delta_rule (std::string const& lhs, std::string const& rhs);
    private:
        std::pair<std::string, std::string> pattern;
    };

    rule_ptr make_delta (std::string const& lhs, std::string const& rhs);

    struct distinct_rule : public index_rule {
        virtual bool operator() (indices_t const& i_ind, indices_t const& j_ind) const override;
        virtual rule_ptr clone () const override;
    };

    rule_ptr make_distinct ();

    struct contraction {
        double operator() (indices_t const& i_ind, indices_t const& j_ind) const;
        contraction (double, rule_ptr &&);
        contraction (contraction const&);
    private:
        double weight;
        rule_ptr rule_;
    };

    struct tensor_factory {
        tensor_factory (std::vector<contraction> && bc, std::vector<contraction> && cc);

        template <typename Confpol>
        boost::multi_array<double, 2> get (Confpol const& cpol) const {
            symmetry_policy::none symm;
            size_t rank = cpol.rank();
            size_t size = symm.size(cpol.range(), rank);
            boost::multi_array<double, 2> res(boost::extents[size][size]);
            indices_t i_ind(rank);
            for (auto row : res) {
                indices_t i_ind_block = cpol.block_indices(i_ind);
                indices_t i_ind_component = cpol.component_indices(i_ind);
                indices_t j_ind(rank);
                for (auto & elem : row) {
                    elem = 0;
                    indices_t j_ind_block = cpol.block_indices(j_ind);
                    indices_t j_ind_component = cpol.component_indices(j_ind);
                    for (auto const& bc : block_contractions)
                        for (auto const& cc : component_contractions)
                            elem += (bc(i_ind_block, j_ind_block)
                                     * cc(i_ind_component, j_ind_component));
                    symm.advance_ind(j_ind, cpol.range());
                }
                symm.advance_ind(i_ind, cpol.range());
            }
            return res;
        }
    private:
        std::vector<contraction> block_contractions;
        std::vector<contraction> component_contractions;
    };

    const std::map<std::string, tensor_factory> exact_tensor = {
        {
            "Cinfv",
            {
                {
                    {  -1, make_delta("a", "b")},
                    {   2, make_delta("a", "a")}
                },
                {
                    {   1, make_delta("a", "a")}
                }
            }
        },
        {
            "Dinfh",
            {
                {
                    {   1./9, make_delta("aa", "bb")},
                    {  -1./3, make_delta("aa", "22")},
                    {  -1./3, make_delta("22", "aa")},
                    {     1., make_delta("22", "22")}
                },
                {
                    {     1., make_delta("ab", "ab")},
                    {     1., make_delta("ab", "ba")},
                    {  -2./3, make_delta("aa", "bb")}
                }
            }
        },
        {
            "Td",
            {
                {
                    {     1.,  make_distinct()}
                },
                {
                    {     1.,  make_delta("abc", "abc")},
                    {     1.,  make_delta("abc", "bca")},
                    {     1.,  make_delta("abc", "cab")},
                    {     1.,  make_delta("abc", "bac")},
                    {     1.,  make_delta("abc", "acb")},
                    {     1.,  make_delta("abc", "cba")},
                    {  -2./5,  make_delta("aac", "bbc")},
                    {  -2./5,  make_delta("aac", "bcb")},
                    {  -2./5,  make_delta("aac", "cbb")},
                    {  -2./5,  make_delta("aca", "bbc")},
                    {  -2./5,  make_delta("aca", "bcb")},
                    {  -2./5,  make_delta("aca", "cbb")},
                    {  -2./5,  make_delta("caa", "bbc")},
                    {  -2./5,  make_delta("caa", "bcb")},
                    {  -2./5,  make_delta("caa", "cbb")}
                }
            }
        }
    };

}
}
