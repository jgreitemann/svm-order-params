#include "point_groups.hpp"

void point_groups::determine_symmetry(std::string str1, int size, bool hds, Rt_array& arr1) 
{
	
	if(str1 == "T" && size == 12)
	{ 
		generate_T();
		arr1.resize(boost::extents[size]);
		arr1 = T_bath;
		return;
	} 

	 else if(str1 == "Cinfv" && size == 1 && hds)
		 {
		 	generate_Cinfv();
			arr1.resize(boost::extents[size]);
			arr1 = Cinfv_bath;
			return;
		 }	

	else if(str1 == "Dinfh" && size == 2 && hds)
		 {
		 	generate_Dinfh();
			arr1.resize(boost::extents[size]);
			arr1 = Dinfh_bath;
			return;
		 }	

	else if(str1 == "Td" && size == 24 && hds) 
		 { 
			generate_Td();
			arr1.resize(boost::extents[size]);
			arr1 = Td_bath;
			return;
		 }
	
	else if(str1 == "Th" && size == 24 && hds) 
		 { 
			generate_Th();
			arr1.resize(boost::extents[size]);
			arr1 = Th_bath;
			return;
		 }

	else if(str1 == "O" && size == 24) 
		 { 
			generate_O();
			arr1.resize(boost::extents[size]);
			arr1 = O_bath;
			return;
		 }

		 else if(str1 == "Oh" && size == 48 && hds)
		 {
		 	generate_Oh();
			arr1.resize(boost::extents[size]);
			arr1 = Oh_bath;
			return;
		 }

		 else if(str1 == "I" && size == 60)
		 {
			generate_I();
			arr1.resize(boost::extents[size]);
			arr1 = I_bath;
			return;
		 }

		 else if(str1 == "Ih" && size == 120 && hds)
		 {
		 	generate_Ih();
			arr1.resize(boost::extents[size]);
			arr1 = Ih_bath;
			return;
		 }

			else 
			{
				std::cout << " \n UNKNOWN symmetry or ERROR in its size or handedness \n" << std::endl;
				return;
			}

}


void point_groups::generate_D2() {
	
	D2_bath.resize(boost::extents[4]);
	
	
	D2_bath[1] << -1, 0, 0,
				0, -1, 0,
				0, 0, 1;
	
	D2_bath[2] << -1, 0, 0,
				0, 1, 0,
				0, 0, -1;
	
	D2_bath[3] = D2_bath[1] * D2_bath[2];
					
	D2_bath[0] = D2_bath[1] * D2_bath[1];
	
	}

void point_groups::generate_Cinfv() { // ignore the inplange elements, only works when J1 = 0
	Cinfv_bath.resize(boost::extents[1]);
	Cinfv_bath[0] = Eigen::MatrixXd::Identity(3,3); 
}	
	
void point_groups::generate_Dinfh() { // as Cinfv
	Dinfh_bath.resize(boost::extents[2]);
	Dinfh_bath[0] = Eigen::MatrixXd::Identity(3,3);
	Dinfh_bath[1] = - Dinfh_bath[0];
}

void point_groups::generate_T() 
{
	
	/* use D2 and c3 as generating groups */
	generate_D2();
	int len_D2 = D2_bath.shape()[0];

	/* define c3 of axis (1,1,1) */
	Rt_array c3;
	c3.resize(boost::extents[3]);
	int len_c3 = c3.shape()[0];
	
	c3[1] << 0, 0, 1,
			 1, 0, 0,
			 0, 1, 0;
	
	c3[2] = c3[1] * c3[1];	
	c3[0] = c3[1] * c3[2];
	

	int len_T = 12;
	T_bath.resize(boost::extents[len_T]);
	
	/* generate T by coset decomposition;
	 * checked */
	int k = 0;
	for(int j = 0; j < len_c3; j++)
		for (int i = 0; i < len_D2; i ++)
			{
				T_bath[k] = c3[j] * D2_bath[i]; 
				k++; 
			}		
	
}

void point_groups::generate_Td()
{
	generate_T();
	int len_T = 12;

	int len_Td = 24;
	Td_bath.resize(boost::extents[len_Td]);

	Rt_array Z2;
	int len_Z2 = 2;
	Z2.resize(boost::extents[len_Z2]);

	Z2[0] << 1, 0, 0,
			 0, 1, 0,
			 0, 0, 1;

	/* m x x z*/		 
	Z2[1] << 0, 1, 0, 
  			 1, 0, 0,  
  			 0, 0, 1; 

	int k = 0;
	for(int j = 0; j < len_Z2; j++)
		for(int i = 0; i < len_T; i++)
		{
			Td_bath[k] = Z2[j] * T_bath[i];
			k++;
		}
}

void point_groups::generate_Th()
{
	generate_T();
	int len_T = 12;

	int len_Th = 24;
	Th_bath.resize(boost::extents[len_Th]);

	Rt_array Z2;
	int len_Z2 = 2;
	Z2.resize(boost::extents[len_Z2]);

	Z2[0] << 1, 0, 0,
			 0, 1, 0,
			 0, 0, 1;

	/* -1 */		 
	Z2[1] << -1, 0, 0, 
  			 0, -1, 0,  
  			 0, 0, -1; 

	int k = 0;
	for(int j = 0; j < len_Z2; j++)
		for(int i = 0; i < len_T; i++)
		{
			Th_bath[k] = Z2[j] * T_bath[i];
			k++;
		}
}

void point_groups::generate_O()	{
	/* use T and a Z2 as generating groups; checked*/
	generate_T();
	int len_T = 12;

	int len_O = 24;
	O_bath.resize(boost::extents[len_O]);

	Rt_array Z2;
	int len_Z2 = 2;
	Z2.resize(boost::extents[len_Z2]);

	Z2[0] << 1, 0, 0,
			 0, 1, 0,
			 0, 0, 1;

	/*2 x,x,0; leads to C4(z) with C2(y)*/		 
	Z2[1] << 0, 1, 0, 
  			 1, 0, 0,  
  			 0, 0, -1; 

	int k = 0;
	for(int j = 0; j < len_Z2; j++)
		for(int i = 0; i < len_T; i++)
		{
			O_bath[k] = Z2[j] * T_bath[i];
			k++;
		}


}

void point_groups::generate_Oh()
{
	generate_O();
	int len_O = 24;

	int len_Oh = 48;
	Oh_bath.resize(boost::extents[len_Oh]);

	Rt_array Z2;
	int len_Z2 = 2;
	Z2.resize(boost::extents[len_Z2]);

	Z2[0] << 1, 0, 0,
			 0, 1, 0,
			 0, 0, 1;

	Z2[1] << -1, 0, 0, 
  			 0, -1, 0,  
  			 0, 0, -1; 

	int k = 0;
	for(int j = 0; j < len_Z2; j++)
		for(int i = 0; i < len_O; i++)
		{
			Oh_bath[k] = Z2[j] * O_bath[i];
			k++;
		}
}

void point_groups::generate_I()
{
	generate_T();
	int len_T = 12;

	/* define c5 */
	Rt_array c5;
	c5.resize(boost::extents[5]);
	int len_c5 = c5.shape()[0];

	c5[1] << 0.5, -0.5*gr, 0.5/gr,
			 0.5*gr, 0.5/gr, -0.5,
			 0.5/gr, 0.5, 0.5*gr;

	c5[2] = c5[1] * c5[1];
	c5[3] = c5[2] * c5[1];
	c5[4] = c5[3] * c5[1];
	c5[0] << 1, 0, 0,
			 0, 1, 0,
			 0, 0, 1;


	int len_I = len_c5 * len_T;
	I_bath.resize(boost::extents[len_I]);

	int k = 0;
	for(int j = 0; j < len_c5; j++)
		for(int i = 0; i < len_T; i++)
		{
			I_bath[k] = c5[j] * T_bath[i];
			k++;
		}

}

void point_groups::generate_Ih()
{
	generate_I();
	int len_I = 60;

	int len_Ih = 120;
	Ih_bath.resize(boost::extents[len_Ih]);

	Rt_array Z2;
	int len_Z2 = 2;
	Z2.resize(boost::extents[len_Z2]);

	Z2[0] << 1, 0, 0,
			 0, 1, 0,
			 0, 0, 1;
	 
	Z2[1] << -1, 0, 0, 
  			 0, -1, 0,  
  			 0, 0, -1; 

	int k = 0;
	for(int j = 0; j < len_Z2; j++)
		for(int i = 0; i < len_I; i++)
		{
			Ih_bath[k] = Z2[j] * I_bath[i];
			k++;
		}
}