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

#include "argh.h"
#include "config_policy.hpp"
#include "contraction.hpp"
#include "matrix_output.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <boost/multi_array.hpp>


int main(int argc, char** argv) {
    argh::parser cmdl;
    cmdl.parse(argc, argv, argh::parser::SINGLE_DASH_IS_MULTIFLAG);

    using ElementPolicy = element_policy::components;
    using confpol_t = block_config_policy<symmetry_policy::none,
                                          ElementPolicy>;
    size_t rank;
    if (!(cmdl(1) >> rank))
        throw std::runtime_error("invalid rank '" + cmdl[1] + "'");

    size_t n_components;
    if (!(cmdl({"-c", "--components"}) >> n_components))
        n_components = 3;
    confpol_t confpol(rank, ElementPolicy{n_components}, false);

    auto contractions = get_contractions(rank);
    auto block = confpol.all_block_indices().begin()->second;

    boost::multi_array<double, 2> c(boost::extents[block.size()][block.size()]);
    auto c_it = contractions.begin();
    for (size_t k = 0; k < contractions.size(); ++k, ++c_it) {
        bool so = c_it->is_self_contraction();
        if (!((so && cmdl[{"-s", "--self"}]) || (!so && cmdl[{"-o", "--outer"}])))
            continue;
        std::string cname = [&] {
            std::stringstream ss;
            ss << *c_it;
            return ss.str();
        } ();

#pragma omp parallel for
        for (size_t i = 0; i < block.size(); ++i)
            for (size_t j = 0; j < block.size(); ++j)
                c[i][j] = contractions[k](block[i].second, block[j].second) ? 1 : 0;

        write_matrix(c, cname, color::grayscale.rescale(1, 0));
    }

}
