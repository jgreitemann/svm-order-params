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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include <lattice/chain.hpp>

#include <algorithm>
#include <iterator>
#include <vector>

struct int_site {
    int i;
};

template <typename Lattice>
void test_nn(Lattice && l, int nn_ind[][Lattice::coordination]) {
    int i = 0;
    for (auto it = l.begin(); it != l.end(); ++it, ++i) {
        CHECK(it->i == i);
        auto nn = l.nearest_neighbors(it);
        std::vector<int> nni;
        std::transform(nn.begin(), nn.end(), std::back_inserter(nni),
                       [] (auto it_n) { return it_n->i; });
        std::sort(nni.begin(), nni.end());
        CHECK(std::equal(std::begin(nn_ind[i]), std::end(nn_ind[i]),
                         nni.begin(), nni.end()));
    }
}

TEST_CASE("nn-chain") {
    {
        int nn_ind[][2] = {
            {1, 1},
            {0, 0}
        };
        test_nn(lattice::chain<int_site>(2, true,
                                         [i=0]() mutable -> int_site { return {i++}; }),
                nn_ind);
    }
    {
        int nn_ind[][2] = {
            {1, 4},
            {0, 2},
            {1, 3},
            {2, 4},
            {0, 3}
        };
        test_nn(lattice::chain<int_site>(5, true,
                                         [i=0]() mutable -> int_site { return {i++}; }),
                nn_ind);
    }
}
