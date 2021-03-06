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

#include <algorithm>
#include <iterator>
#include <set>
#include <stdexcept>

#include <tksvm/utilities/results/results.hpp>


using namespace tksvm::results;

delta_rule::delta_rule (std::string const& lhs, std::string const& rhs)
    : pattern {lhs, rhs} {}

bool delta_rule::operator() (indices_t const& i_ind, indices_t const& j_ind) const {
    if (i_ind.size() != pattern.first.size() || j_ind.size() != pattern.second.size())
        throw std::runtime_error("index ranks don't match delta rule pattern");

    std::map<char, size_t> assignments;
    auto idx_it = i_ind.begin();
    auto c_it = pattern.first.begin();
    while (idx_it != j_ind.end()) {
        if ('0' <= *c_it && *c_it <= '9') {
            if (size_t(*c_it - '0') != *idx_it)
                return false;
        } else {
            auto find = assignments.find(*c_it);
            if (find == assignments.end()) {
                assignments[*c_it] = *idx_it;
            } else {
                if (find->second != *idx_it)
                    return false;
            }
        }

        ++idx_it;
        ++c_it;
        if (idx_it == i_ind.end()) {
            idx_it = j_ind.begin();
            c_it = pattern.second.begin();
        }
    }
    return true;
}

rule_ptr delta_rule::clone () const {
    return rule_ptr(new delta_rule(*this));
}

rule_ptr tksvm::results::make_delta (std::string const& lhs, std::string const& rhs) {
    return rule_ptr(new delta_rule(lhs, rhs));
}

bool distinct_rule::operator() (indices_t const& i_ind, indices_t const& j_ind) const {
    std::set<size_t> set(i_ind.begin(), i_ind.end());
    if (i_ind.size() != set.size()) return false;
    set.clear();
    set.insert(j_ind.begin(), j_ind.end());
    return j_ind.size() == set.size();
}

rule_ptr distinct_rule::clone () const {
    return rule_ptr(new distinct_rule(*this));
}

rule_ptr tksvm::results::make_distinct () {
    return rule_ptr(new distinct_rule());
}

contraction::contraction (double weight, rule_ptr && rule)
    : weight(weight), rule_(std::forward<rule_ptr>(rule)) {}

contraction::contraction (contraction const& other)
    : weight(other.weight), rule_(other.rule_->clone()) {}

double contraction::operator() (indices_t const& i_ind, indices_t const& j_ind) const {
    return (*rule_)(i_ind, j_ind) ? weight : 0.;
}

tensor_factory::tensor_factory (std::vector<contraction> && bc,
                                std::vector<contraction> && cc)
    : block_contractions(std::forward<std::vector<contraction>>(bc)),
      component_contractions(std::forward<std::vector<contraction>>(cc)) {}
