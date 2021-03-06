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

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <tksvm/utilities/contraction.hpp>
#include <tksvm/utilities/indices.hpp>


bool tksvm::contraction::is_self_contraction () const {
    for (auto ep : endpoints)
        if (ep < endpoints.size())
            return true;
    return false;
}

bool tksvm::contraction::operator() (indices_t const& i_ind,
                                     indices_t const& j_ind) const
{
    std::vector<bool> checked(2 * endpoints.size(), false);
    auto ind_get = [&] (size_t i) {
        return i >= i_ind.size() ? j_ind[i - i_ind.size()] : i_ind[i];
    };
    auto it = endpoints.begin();
    for (size_t i = 0; i < 2 * endpoints.size(); ++i) {
        if (checked[i])
            continue;
        if (ind_get(i) != ind_get(*it))
            return false;
        checked[*it] = true;
        ++it;
    }
    return true;
}

std::ostream & tksvm::operator<< (std::ostream & os, contraction const& ct) {
    char c = 'a';
    std::string out(2 * ct.endpoints.size(), '-');
    auto it = ct.endpoints.begin();
    for (auto & o : out) {
        if (o == '-') {
            o = c++;
            out[*(it++)] = o;
        }
    }
    return os << '[' << out.substr(0, ct.endpoints.size())
              << ';' << out.substr(ct.endpoints.size(), ct.endpoints.size())
              << ']';
}

std::vector<tksvm::contraction> tksvm::get_contractions(size_t rank) {
    // define a recursive lambda
    static void (*rec)(std::vector<tksvm::contraction> &,
                       std::vector<size_t>,
                       std::set<size_t>)
        = [] (std::vector<tksvm::contraction> & cs,
              std::vector<size_t> ep,
              std::set<size_t> pl)
        {
            if (pl.size() == 2) {
                ep.push_back(*(++pl.begin()));
                cs.push_back(std::move(ep));
            } else {
                pl.erase(pl.begin());
                std::vector<size_t> epn(ep);
                for(auto it = pl.begin(); it != pl.end(); ++it) {
                    epn.push_back(*it);
                    pl.erase(it);
                    rec(cs, epn, pl);
                    it = pl.insert(epn.back()).first;
                    epn.pop_back();
                }
            }
        };

    std::vector<tksvm::contraction> cs;
    std::set<size_t> places;
    for (size_t i = 0; i < 2 * rank; ++i)
        places.insert(i);

    rec(cs, {}, places);
    return cs;
}

auto tksvm::contraction_matrix(std::vector<contraction> const& cs,
                               index_assoc_vec const& is,
                               index_assoc_vec const& js)
    -> tksvm::contraction_matrix_t
{
    contraction_matrix_t a(is.size() * js.size(), cs.size());
    for (size_t i = 0; i < is.size(); ++i)
        for (size_t j = 0; j < js.size(); ++j)
            for (size_t k = 0; k < cs.size(); ++k)
                a(i * js.size() + j, k) = cs[k](is[i].second, js[j].second) ? 1 : 0;
    return a;
}
