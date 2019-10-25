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

#include <algorithm>
#include <regex>
#include <string>


namespace tksvm {

inline std::string replace_extension(std::string const& old_path,
                                     std::string const& new_ext)
{
	std::regex re{"(.+?)(\\.ini|\\.h5|\\.txt|\\.ppm|\\.out|\\.clone|\\.test)*"};
	std::smatch match;
	if (std::regex_match(old_path, match, re)) {
		return match[1].str() + new_ext;
	} else {
		return old_path + new_ext;
	}
}

}
