// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2019  Jonas Greitemann, Ke Liu, and Lode Pollet

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

template <typename Config>
struct config_serializer {
	using config_t = Config;
	template <typename OutputIterator>
	void serialize(config_t const& conf, OutputIterator it) const {
		std::copy(conf.begin(), conf.end(), it);
	}

	template <typename InputIterator>
	void deserialize(InputIterator it, config_t & conf) const {
		std::copy_n(it, std::distance(conf.begin(), conf.end()), conf.begin());
	}
};