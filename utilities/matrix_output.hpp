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

#include "colormap.hpp"
#include "filesystem.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

#include <boost/multi_array.hpp>


inline double normalize_matrix (boost::multi_array<double,2> & mat) {
    double absmax = 0.;
    for (size_t i = 0; i < mat.shape()[0]; ++i)
        for (size_t j = 0; j < mat.shape()[0]; ++j)
            if (std::abs(absmax) < std::abs(mat[i][j]))
                absmax = mat[i][j];
    for (size_t i = 0; i < mat.shape()[0]; ++i)
        for (size_t j = 0; j < mat.shape()[0]; ++j)
            mat[i][j] /= absmax;
    return absmax;
}

template <typename Palette>
inline void write_matrix (boost::multi_array<double,2> const& mat,
                          std::string basename,
                          Palette const& pal)
{
    /* PPM output */ {
        std::array<size_t, 2> shape {mat.shape()[0], mat.shape()[1]};
        typedef boost::multi_array_types::index_range range;
        boost::multi_array<double,2> flat(mat[boost::indices[range(shape[0]-1, -1, -1)][range(0, shape[1])]]);
        flat.reshape(std::array<size_t,2>{1, flat.num_elements()});
        auto pixit = itadpt::map_iterator(flat[0].begin(), pal);
        auto pmap = color::pixmap<decltype(pixit)>(pixit, shape);

        std::ofstream ppm(basename + "." + pmap.file_extension(), std::ios::binary);
        pmap.write_binary(ppm);
    }
    /* TXT output */ {
        std::ofstream txt(basename + ".txt");
        for (auto row_it = mat.rbegin(); row_it != mat.rend(); ++row_it) {
            for (double elem : *row_it) {
                txt << elem << '\t';
            }
            txt << '\n';
        }
    }
}
