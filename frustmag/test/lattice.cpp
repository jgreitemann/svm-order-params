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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest.h"
#include <lattice/chain.hpp>
#include <lattice/ortho.hpp>
#include <lattice/triangular.hpp>
#include <lattice/honeycomb.hpp>
#include <lattice/kagome.hpp>
#include <lattice/dice.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

struct int_site {
    int i;
};

static_assert(!is_archivable<int_site>::value,
              "int_site is_archivable when it shouldn't be.");
static_assert(!is_serializable<int_site>::value,
              "int_site is_serializable when it shouldn't be.");

template <typename Lattice>
void test_nn(Lattice && l, int nn_ind[][Lattice::coordination]) {
    int i = 0;
    for (auto it = l.begin(); it != l.end(); ++it, ++i) {
        CHECK(it->i == i);
        auto nn = l.nearest_neighbors(it);
        std::vector<int> nni;
        std::transform(nn.begin(), nn.end(), std::back_inserter(nni),
                       [end=l.end()] (auto it_n) {
                           return it_n == end ? -1 : it_n->i;
                       });
        std::sort(nni.begin(), nni.end());
        bool nn_equal = std::equal(std::begin(nn_ind[i]), std::end(nn_ind[i]),
                                   nni.begin(), nni.end());
        if (!nn_equal) {
            std::copy(std::begin(nn_ind[i]), std::end(nn_ind[i]),
                      std::ostream_iterator<int>{std::cout, ", "});
            std::cout << std::endl;
            std::copy(nni.begin(), nni.end(),
                      std::ostream_iterator<int>{std::cout, ", "});
            std::cout << std::endl;
            
        }
        CHECK(nn_equal);
    }
}

auto increment_gen = [] {
    return [i=0]() mutable -> int_site { return {i++}; };
};

TEST_CASE("nn-chain-periodic-2") {
    int nn_ind[][2] = {
        {1, 1},
        {0, 0}
    };
    test_nn(lattice::chain<int_site>(2, true, increment_gen()), nn_ind);
}

TEST_CASE("nn-chain-open-2") {
    int nn_ind[][2] = {
        {-1, 1},
        {-1, 0}
    };
    test_nn(lattice::chain<int_site>(2, false, increment_gen()), nn_ind);
}

TEST_CASE("nn-chain-periodic-5") {
    int nn_ind[][2] = {
        {1, 4},
        {0, 2},
        {1, 3},
        {2, 4},
        {0, 3}
    };
    test_nn(lattice::chain<int_site>(5, true, increment_gen()), nn_ind);
}

TEST_CASE("nn-chain-open-5") {
    int nn_ind[][2] = {
        {-1, 1},
        { 0, 2},
        { 1, 3},
        { 2, 4},
        {-1, 3}
    };
    test_nn(lattice::chain<int_site>(5, false, increment_gen()), nn_ind);
}

TEST_CASE("nn-square-periodic-2") {
    int nn_ind[][4] = {
        {1, 1, 2, 2},
        {0, 0, 3, 3},
        {0, 0, 3, 3},
        {1, 1, 2, 2}
    };
    test_nn(lattice::square<int_site>(2, true, increment_gen()), nn_ind);
}

TEST_CASE("nn-square-open-2") {
    int nn_ind[][4] = {
        {-1, -1, 1, 2},
        {-1, -1, 0, 3},
        {-1, -1, 0, 3},
        {-1, -1, 1, 2},
    };
    test_nn(lattice::square<int_site>(2, false, increment_gen()), nn_ind);
}

TEST_CASE("nn-square-periodic-4") {
    int nn_ind[][4] = {
        { 1,  3,  4, 12},
        { 0,  2,  5, 13},
        { 1,  3,  6, 14},
        { 0,  2,  7, 15},
        { 0,  5,  7,  8},
        { 1,  4,  6,  9},
        { 2,  5,  7, 10},
        { 3,  4,  6, 11},
        { 4,  9, 11, 12},
        { 5,  8, 10, 13},
        { 6,  9, 11, 14},
        { 7,  8, 10, 15},
        { 0,  8, 13, 15},
        { 1,  9, 12, 14},
        { 2, 10, 13, 15},
        { 3, 11, 12, 14},
    };
    test_nn(lattice::square<int_site>(4, true, increment_gen()), nn_ind);
}

TEST_CASE("nn-square-open-4") {
    int nn_ind[][4] = {
        {-1, -1,  1,  4},
        {-1,  0,  2,  5},
        {-1,  1,  3,  6},
        {-1, -1,  2,  7},
        {-1,  0,  5,  8},
        { 1,  4,  6,  9},
        { 2,  5,  7, 10},
        {-1,  3,  6, 11},
        {-1,  4,  9, 12},
        { 5,  8, 10, 13},
        { 6,  9, 11, 14},
        {-1,  7, 10, 15},
        {-1, -1,  8, 13},
        {-1,  9, 12, 14},
        {-1, 10, 13, 15},
        {-1, -1, 11, 14},
    };
    test_nn(lattice::square<int_site>(4, false, increment_gen()), nn_ind);
}

TEST_CASE("nn-cubic-periodic-3") {
    int nn_ind[][6] = {
        { 1,  2,  3,  6,  9, 18},
        { 0,  2,  4,  7, 10, 19},
        { 0,  1,  5,  8, 11, 20},
        { 0,  4,  5,  6, 12, 21},
        { 1,  3,  5,  7, 13, 22},
        { 2,  3,  4,  8, 14, 23},
        { 0,  3,  7,  8, 15, 24},
        { 1,  4,  6,  8, 16, 25},
        { 2,  5,  6,  7, 17, 26},
        { 0, 10, 11, 12, 15, 18},
        { 1,  9, 11, 13, 16, 19},
        { 2,  9, 10, 14, 17, 20},
        { 3,  9, 13, 14, 15, 21},
        { 4, 10, 12, 14, 16, 22},
        { 5, 11, 12, 13, 17, 23},
        { 6,  9, 12, 16, 17, 24},
        { 7, 10, 13, 15, 17, 25},
        { 8, 11, 14, 15, 16, 26},
        { 0,  9, 19, 20, 21, 24},
        { 1, 10, 18, 20, 22, 25},
        { 2, 11, 18, 19, 23, 26},
        { 3, 12, 18, 22, 23, 24},
        { 4, 13, 19, 21, 23, 25},
        { 5, 14, 20, 21, 22, 26},
        { 6, 15, 18, 21, 25, 26},
        { 7, 16, 19, 22, 24, 26},
        { 8, 17, 20, 23, 24, 25},
    };
    test_nn(lattice::cubic<int_site>(3, true, increment_gen()), nn_ind);
}

TEST_CASE("nn-cubic-open-3") {
    int nn_ind[][6] = {
        {-1, -1, -1,  1,  3,  9},
        {-1, -1,  0,  2,  4, 10},
        {-1, -1, -1,  1,  5, 11},
        {-1, -1,  0,  4,  6, 12},
        {-1,  1,  3,  5,  7, 13},
        {-1, -1,  2,  4,  8, 14},
        {-1, -1, -1,  3,  7, 15},
        {-1, -1,  4,  6,  8, 16},
        {-1, -1, -1,  5,  7, 17},
        {-1, -1,  0, 10, 12, 18},
        {-1,  1,  9, 11, 13, 19},
        {-1, -1,  2, 10, 14, 20},
        {-1,  3,  9, 13, 15, 21},
        { 4, 10, 12, 14, 16, 22},
        {-1,  5, 11, 13, 17, 23},
        {-1, -1,  6, 12, 16, 24},
        {-1,  7, 13, 15, 17, 25},
        {-1, -1,  8, 14, 16, 26},
        {-1, -1, -1,  9, 19, 21},
        {-1, -1, 10, 18, 20, 22},
        {-1, -1, -1, 11, 19, 23},
        {-1, -1, 12, 18, 22, 24},
        {-1, 13, 19, 21, 23, 25},
        {-1, -1, 14, 20, 22, 26},
        {-1, -1, -1, 15, 21, 25},
        {-1, -1, 16, 22, 24, 26},
        {-1, -1, -1, 17, 23, 25},
    };
    test_nn(lattice::cubic<int_site>(3, false, increment_gen()), nn_ind);
}

TEST_CASE("nn-triangular-periodic-3") {
    int nn_ind[][6] = {
        {1, 2, 3, 5, 6, 7},
        {0, 2, 3, 4, 7, 8},
        {0, 1, 4, 5, 6, 8},
        {0, 1, 4, 5, 6, 8},
        {1, 2, 3, 5, 6, 7},
        {0, 2, 3, 4, 7, 8},
        {0, 2, 3, 4, 7, 8},
        {0, 1, 4, 5, 6, 8},
        {1, 2, 3, 5, 6, 7},
    };
    test_nn(lattice::triangular<int_site>(3, true, increment_gen()), nn_ind);
}

TEST_CASE("nn-triangular-open-3") {
    int nn_ind[][6] = {
        {-1, -1, -1, -1,  1,  3},
        {-1, -1,  0,  2,  3,  4},
        {-1, -1, -1,  1,  4,  5},
        {-1, -1,  0,  1,  4,  6},
        { 1,  2,  3,  5,  6,  7},
        {-1, -1,  2,  4,  7,  8},
        {-1, -1, -1,  3,  4,  7},
        {-1, -1,  4,  5,  6,  8},
        {-1, -1, -1, -1,  5,  7},
    };
    test_nn(lattice::triangular<int_site>(3, false, increment_gen()), nn_ind);
}

TEST_CASE("nn-triangular-periodic-5") {
    int nn_ind[][6] = {
        { 1,  4,  5,  9, 20, 21},
        { 0,  2,  5,  6, 21, 22},
        { 1,  3,  6,  7, 22, 23},
        { 2,  4,  7,  8, 23, 24},
        { 0,  3,  8,  9, 20, 24},
        { 0,  1,  6,  9, 10, 14},
        { 1,  2,  5,  7, 10, 11},
        { 2,  3,  6,  8, 11, 12},
        { 3,  4,  7,  9, 12, 13},
        { 0,  4,  5,  8, 13, 14},
        { 5,  6, 11, 14, 15, 19},
        { 6,  7, 10, 12, 15, 16},
        { 7,  8, 11, 13, 16, 17},
        { 8,  9, 12, 14, 17, 18},
        { 5,  9, 10, 13, 18, 19},
        {10, 11, 16, 19, 20, 24},
        {11, 12, 15, 17, 20, 21},
        {12, 13, 16, 18, 21, 22},
        {13, 14, 17, 19, 22, 23},
        {10, 14, 15, 18, 23, 24},
        { 0,  4, 15, 16, 21, 24},
        { 0,  1, 16, 17, 20, 22},
        { 1,  2, 17, 18, 21, 23},
        { 2,  3, 18, 19, 22, 24},
        { 3,  4, 15, 19, 20, 23},
    };
    test_nn(lattice::triangular<int_site>(5, true, increment_gen()), nn_ind);
}

TEST_CASE("nn-honeycomb-periodic-3") {
    int nn_ind[][3] = {
        { 1,  5, 13},
        { 0,  2,  6},
        { 1,  3, 15},
        { 2,  4,  8},
        { 3,  5, 17},
        { 0,  4, 10},
        { 1,  7, 11},
        { 6,  8, 12},
        { 3,  7,  9},
        { 8, 10, 14},
        { 5,  9, 11},
        { 6, 10, 16},
        { 7, 13, 17},
        { 0, 12, 14},
        { 9, 13, 15},
        { 2, 14, 16},
        {11, 15, 17},
        { 4, 12, 16},
    };
    test_nn(lattice::honeycomb<int_site>(3, true, increment_gen()), nn_ind);
}

TEST_CASE("nn-honeycomb-open-3") {
    int nn_ind[][3] = {
        {-1, -1,  1},
        { 0,  2,  6},
        {-1,  1,  3},
        { 2,  4,  8},
        {-1,  3,  5},
        {-1,  4, 10},
        {-1,  1,  7},
        { 6,  8, 12},
        { 3,  7,  9},
        { 8, 10, 14},
        { 5,  9, 11},
        {-1, 10, 16},
        {-1,  7, 13},
        {-1, 12, 14},
        { 9, 13, 15},
        {-1, 14, 16},
        {11, 15, 17},
        {-1, -1, 16},
    };
    test_nn(lattice::honeycomb<int_site>(3, false, increment_gen()), nn_ind);
}

TEST_CASE("nn-kagome-periodic-3") {
    int nn_ind[][4] = {
        { 1,  2,  7, 20},
        { 0,  2,  3, 23},
        { 0,  1,  9, 16},
        { 1,  4,  5, 23},
        { 3,  5,  6, 26},
        { 3,  4, 10, 12},
        { 4,  7,  8, 26},
        { 0,  6,  8, 20},
        { 6,  7, 13, 15},
        { 2, 10, 11, 16},
        { 5,  9, 11, 12},
        { 9, 10, 18, 25},
        { 5, 10, 13, 14},
        { 8, 12, 14, 15},
        {12, 13, 19, 21},
        { 8, 13, 16, 17},
        { 2,  9, 15, 17},
        {15, 16, 22, 24},
        {11, 19, 20, 25},
        {14, 18, 20, 21},
        { 0,  7, 18, 19},
        {14, 19, 22, 23},
        {17, 21, 23, 24},
        { 1,  3, 21, 22},
        {17, 22, 25, 26},
        {11, 18, 24, 26},
        { 4,  6, 24, 25},
    };
    test_nn(lattice::kagome<int_site>(3, true, increment_gen()), nn_ind);
}

TEST_CASE("nn-kagome-open-3") {
    int nn_ind[][4] = {
        {-1, -1,  1,  2},
        {-1,  0,  2,  3},
        {-1,  0,  1,  9},
        {-1,  1,  4,  5},
        {-1,  3,  5,  6},
        { 3,  4, 10, 12},
        {-1,  4,  7,  8},
        {-1, -1,  6,  8},
        { 6,  7, 13, 15},
        {-1,  2, 10, 11},
        { 5,  9, 11, 12},
        {-1,  9, 10, 18},
        { 5, 10, 13, 14},
        { 8, 12, 14, 15},
        {12, 13, 19, 21},
        { 8, 13, 16, 17},
        {-1, -1, 15, 17},
        {15, 16, 22, 24},
        {-1, 11, 19, 20},
        {14, 18, 20, 21},
        {-1, -1, 18, 19},
        {14, 19, 22, 23},
        {17, 21, 23, 24},
        {-1, -1, 21, 22},
        {17, 22, 25, 26},
        {-1, -1, 24, 26},
        {-1, -1, 24, 25},
    };
    test_nn(lattice::kagome<int_site>(3, false, increment_gen()), nn_ind);
}

TEST_CASE("nn-dice-periodic-3") {
    int nn_ind[][6] = {
        { 1,  2,  7,  8, 16, 20},
        {-1, -1, -1,  0,  3, 21},
        {-1, -1, -1,  0,  3,  9},
        { 1,  2,  4,  5, 10, 23},
        {-1, -1, -1,  3,  6, 24},
        {-1, -1, -1,  3,  6, 12},
        { 4,  5,  7,  8, 13, 26},
        {-1, -1, -1,  0,  6, 18},
        {-1, -1, -1,  0,  6, 15},
        { 2, 10, 11, 16, 17, 25},
        {-1, -1, -1,  3,  9, 12},
        {-1, -1, -1,  9, 12, 18},
        { 5, 10, 11, 13, 14, 19},
        {-1, -1, -1,  6, 12, 15},
        {-1, -1, -1, 12, 15, 21},
        { 8, 13, 14, 16, 17, 22},
        {-1, -1, -1,  0,  9, 15},
        {-1, -1, -1,  9, 15, 24},
        { 7, 11, 19, 20, 25, 26},
        {-1, -1, -1, 12, 18, 21},
        {-1, -1, -1,  0, 18, 21},
        { 1, 14, 19, 20, 22, 23},
        {-1, -1, -1, 15, 21, 24},
        {-1, -1, -1,  3, 21, 24},
        { 4, 17, 22, 23, 25, 26},
        {-1, -1, -1,  9, 18, 24},
        {-1, -1, -1,  6, 18, 24},
    };
    test_nn(lattice::dice<int_site>(3, true, increment_gen()), nn_ind);
}

TEST_CASE("nn-dice-open-3") {
    int nn_ind[][6] = {
        {-1, -1, -1, -1,  1,  2},
        {-1, -1, -1, -1,  0,  3},
        {-1, -1, -1,  0,  3,  9},
        {-1,  1,  2,  4,  5, 10},
        {-1, -1, -1, -1,  3,  6},
        {-1, -1, -1,  3,  6, 12},
        {-1,  4,  5,  7,  8, 13},
        {-1, -1, -1, -1, -1,  6},
        {-1, -1, -1, -1,  6, 15},
        {-1, -1, -1,  2, 10, 11},
        {-1, -1, -1,  3,  9, 12},
        {-1, -1, -1,  9, 12, 18},
        { 5, 10, 11, 13, 14, 19},
        {-1, -1, -1,  6, 12, 15},
        {-1, -1, -1, 12, 15, 21},
        { 8, 13, 14, 16, 17, 22},
        {-1, -1, -1, -1, -1, 15},
        {-1, -1, -1, -1, 15, 24},
        {-1, -1, -1, 11, 19, 20},
        {-1, -1, -1, 12, 18, 21},
        {-1, -1, -1, -1, 18, 21},
        {-1, 14, 19, 20, 22, 23},
        {-1, -1, -1, 15, 21, 24},
        {-1, -1, -1, -1, 21, 24},
        {-1, 17, 22, 23, 25, 26},
        {-1, -1, -1, -1, -1, 24},
        {-1, -1, -1, -1, -1, 24},
    };
    test_nn(lattice::dice<int_site>(3, false, increment_gen()), nn_ind);
}
