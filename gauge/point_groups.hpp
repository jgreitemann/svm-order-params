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
#include <boost/multi_array.hpp>
#include "Eigen/Dense"
#include <string>
#include <boost/math/constants/constants.hpp>
#include <iostream>

/*golden ratio*/
constexpr double gr(boost::math::constants::phi<double>());


class point_groups {

	private:
	typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Rt3;
    typedef boost::multi_array<Rt3, 1> Rt_array;

    Rt_array D2_bath;
    Rt_array D2h_bath;
    Rt_array D2d_bath;
    Rt_array D3_bath;
    Rt_array D3h_bath;
    Rt_array Cinfv_bath;
    Rt_array Dinfh_bath;
    Rt_array T_bath;
    Rt_array Td_bath;
    Rt_array Th_bath;
    Rt_array O_bath;
    Rt_array Oh_bath;
    Rt_array I_bath;
    Rt_array Ih_bath;

	public:

	inline void generate_D2();
	inline void generate_D2h();
	inline void generate_D2d();
	inline void generate_D3();
	inline void generate_D3h();
	inline void generate_Cinfv();
	inline void generate_Dinfh();
	inline void generate_T();
	inline void generate_Td();
	inline void generate_Th();
	inline void generate_O();
	inline void generate_Oh();
	inline void generate_I();
	inline void generate_Ih();

	void determine_symmetry(std::string str1, int size, bool hds, Rt_array& arr1);

	};
