// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2017  Jonas Greitemann, Ke Liu, and Lode Pollet

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

#include <experimental/filesystem>


namespace fs = std::experimental::filesystem;


inline std::string replace_extension (std::string const& ini_or_h5, std::string const& new_ext) {
    fs::path ext;
    fs::path known_exts[] = {
        ".ini",
        ".h5",
        ".txt",
        ".ppm",
        ".out",
        ".clone",
        ".test"
    };
    fs::path p(ini_or_h5);
    for (; !(ext = p.extension()).empty(); p = p.stem()) {
        auto it = std::find(std::begin(known_exts), std::end(known_exts), ext);
        if (it == std::end(known_exts))
            break;
    }
    return p += new_ext;
}
