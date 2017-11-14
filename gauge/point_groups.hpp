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
