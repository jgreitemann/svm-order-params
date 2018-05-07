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

#include "gauge.hpp"

//compute glocal order parameters 
// and return the square of the nematicity magnitude; 
//verified at order and disorder limit

	/** Traid-matrix correspondance
	 *	  0				1			2
	 *l= R[i](0,0), R[i](0,1), R[i](0,2)
	 * m = R[i](1,0), R[i](1,1), R[i](1,2)
	 * n = R[i](2,0), R[i](2,1), R[i](2,2)
	 **/

double gauge_sim::nematicity_Cinfv() {
	// Q_a = n_a = R[i](2,a)

	double A[3] = {0}; //sublattice
	double B[3] = {0};
	double Q[3] = {0}; // total
	int i = 0; //initial coordinates in chain

	for(int z=0; z<L; z++)
		for(int y=0; y<L; y++)
			for(int x=0; x<L; x++)
			{
				if((x+y+z)%2 == 0)
				{
					for(int a=0; a<3; a++)
						A[a] += R[i](2,a);
					i++;
				}
				else
				{
					for(int b=0; b<3; b++)
						B[b] += R[i](2,b);
					i++;
				}
			}

	for(int a=0; a<3; a++)
		Q[a] = A[a] + J3*B[a];

	double Q2 = 0;
	
	for(int a=0; a<3; a++)
		Q2 += Q[a]*Q[a];
	
	double norm = L3 * L3;

	return Q2/norm;
}

double gauge_sim::nematicity_Dinfh() { // to or not to take the trace out?
	/* Q_ab = n_a * n_b - 1/3 delta_ab 
			= R[i](2,a) * R[i](2,b) - 1/3 delta_ab
	*/

	double Q[3][3] = {{0}};
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> delta;
	delta = Eigen::MatrixXd::Identity(3,3);

	for(int i = 0; i < L3; i++)
		for(int a = 0; a < 3; a++)
			for(int b = 0; b < 3; b++)
				Q[a][b] += R[i](2,a) * R[i](2,b) - 1.0*delta(a,b)/3;

	double Q2 = 0;
	for(int a = 0; a < 3; a++)
		for(int b = 0; b < 3; b++)
			Q2 += Q[a][b]*Q[a][b];

	double norm = 2./3 * L3 * L3;

	return Q2/norm;
}

double gauge_sim::nematicity_T() 
{

	 /* Intermediate tensor T as part of the order parameter tensor Q */
	 double Q[3][3][3] = {{{0}}};

	 /* Q_abc = l_a m_b n_c  + m_a n_b l_c + n_a l_b m_c 
	 			 - 1/2 epsilon_abc
	 		  = R[i](0, a) R[i](1, b) R[i](2, c)
	 			+ R[i](1, a) R[i](2, b) R[i](0, c)
	 			+ R[i](2, a) R[i](0, b) R[i](1, c)
	 			- 1/2 epsilon_abc 
	 */

	 for(int i = 0; i < L3; i++)
	 	for(int a = 0; a < 3; a++)
	 		for(int b = 0; b < 3; b++)
	 			for(int c = 0; c < 3; c++)
	 				Q[a][b][c] += R[i](0,a) * R[i](1,b) * R[i](2,c)
	 								+ R[i](1,a) * R[i](2,b) * R[i](0,c)
	 								+ R[i](2,a) * R[i](0,b) * R[i](1,c);

	 /* take out the trace fof Q */
	 double halfL3 = L3 * 0.5;

	 Q[0][1][2] -= halfL3; Q[1][2][0] -= halfL3; Q[2][0][1] -= halfL3;
	 Q[1][0][2] += halfL3; Q[2][1][0] += halfL3; Q[0][2][1] += halfL3;

	 /* compute the nematicity; Q2 = 1.5 * L3 at perfect order */	
	 double Q2 = 0;

	 for(int a = 0; a < 3; a++)
	 	for(int b = 0; b < 3; b++)
	 		for(int c = 0; c < 3; c++)
	 			Q2 += Q[a][b][c]*Q[a][b][c];	
	 
	 double norm = 1.5 * L3 * L3;

	 return Q2/norm;
}


double gauge_sim::nematicity_Td() 
{

	double Q[3][3][3] = {{{0}}};

	 /* Q_abc = (l_a m_b + m_a l_b) n_c  + 
	 			(m_a n_b + n_a m_b) l_c + 
	 			(n_a l_b + l_a n_b) m_c 
	 */

	 for(int i = 0; i < L3; i++)
	 	for(int a = 0; a < 3; a++)
	 	for(int b = 0; b < 3; b++)
	 	for(int c = 0; c < 3; c++)
	 		Q[a][b][c] += (R[i](0,a) * R[i](1,b) + 
	 					   R[i](1,a) * R[i](0,b)) * R[i](2,c)
	 					+ (R[i](1,a) * R[i](2,b) + 
	 					   R[i](2,a) * R[i](1,b)) * R[i](0,c)
	 					 + (R[i](2,a) * R[i](0,b) +
	 					    R[i](0,a) * R[i](2,b)) * R[i](1,c);

	 /* compute the nematicity; Q2 = 1.5 * L3 at perfect order */	
	 double Q2 = 0;

	 for(int a = 0; a < 3; a++)
	 	for(int b = 0; b < 3; b++)
	 	for(int c = 0; c < 3; c++)
	 		Q2 += Q[a][b][c]*Q[a][b][c];	
	 
	 double norm = 6.0 * L3 * L3;

	 return Q2/norm;
}

double gauge_sim::nematicity_Th() 
{
	 double Q[3][3][3][3] = {{{{0}}}};
	 double trace_Th[3][3][3][3] = {{{{0}}}};

	 Eigen::Matrix<double, 3, 3, Eigen::RowMajor> delta;
	 delta = Eigen::MatrixXd::Identity(3,3);

	 /*define the trace*/
	 for(int a = 0; a < 3; a++)
	 	for(int b = 0; b < 3; b++)
	 	for(int c = 0; c < 3; c++)
	 	for(int d = 0; d < 3; d++)
	 		trace_Th[a][b][c][d] = 0.4 * delta(a,b) * delta(c,d) - 0.1 * (delta(a,c) * delta(b,d) + delta(a,d) * delta(b,c));

	 for(int i = 0; i < L3; i++)
	 	for(int a = 0; a < 3; a++)
	 	for(int b = 0; b < 3; b++)
	 	for(int c = 0; c < 3; c++)
	 	for(int d = 0; d < 3; d++)
	 		Q[a][b][c][d] +=
	 			R[i](0,a) * R[i](0,b) * R[i](1,c) * R[i](1,d)
	  			+ R[i](1,a) * R[i](1,b) * R[i](2,c) * R[i](2,d)
	  			+ R[i](2,a) * R[i](2,b) * R[i](0,c) * R[i](0,d) 
	  			- trace_Th[a][b][c][d];

	 /* compute the nematicity; Q2 = 1.2 * L3 * L3 at perfect order */	
	 double Q2 = 0;

	 for(int a = 0; a < 3; a++)
	 	for(int b = 0; b < 3; b++)
	 	for(int c = 0; c < 3; c++)
	 	for(int d = 0; d < 3; d++)
			Q2 += Q[a][b][c][d]*Q[a][b][c][d];	
	
	 
	 double norm = 1.8 * L3 * L3;

	 return Q2/norm;
}

double gauge_sim::nematicity_Oh() 
{
	 /* Q_abcd 
	 =  l_a l_b l_c l_d  + m_a m_b m_c m_d + n_a n_b n_c n_d - 1/5 sum delta_ab delta_cd
	 = R[i](0,a) R[i](0,b) R[i](0,c) R[i](0,d)
	  + R[i](1,a) R[i](1,b) R[i](1,c) R[i](1,d)
	  + R[i](2,a) R[i](2,b) R[i](2,c) R[i](2,d)
	  - 1/5 sum delta_ab delta_cd
	 */
	 double Q[3][3][3][3] = {{{{0}}}};
	 double trace_Oh[3][3][3][3] = {{{{0}}}};

	 Eigen::Matrix<double, 3, 3, Eigen::RowMajor> delta;
	 delta = Eigen::MatrixXd::Identity(3,3);

	 /*define the trace*/
	 for(int a = 0; a < 3; a++)
	 	for(int b = 0; b < 3; b++)
	 	for(int c = 0; c < 3; c++)
	 	for(int d = 0; d < 3; d++)
	 		trace_Oh[a][b][c][d] = 0.2 * ( delta(a,b) * delta(c,d) + delta(a,c) * delta(b,d) + delta(a,d) * delta(b,c) );

	 for(int i = 0; i < L3; i++)
	 	for(int a = 0; a < 3; a++)
	 	for(int b = 0; b < 3; b++)
	 	for(int c = 0; c < 3; c++)
	 	for(int d = 0; d < 3; d++)
	 		Q[a][b][c][d] +=
	 			R[i](0,a) * R[i](0,b) * R[i](0,c) * R[i](0,d)
	  			+ R[i](1,a) * R[i](1,b) * R[i](1,c) * R[i](1,d)
	  			+ R[i](2,a) * R[i](2,b) * R[i](2,c) * R[i](2,d) 
	  			- trace_Oh[a][b][c][d];

	 /* compute the nematicity; Q2 = 1.2 * L3 * L3 at perfect order */	
	 double Q2 = 0;

	 for(int a = 0; a < 3; a++)
	 	for(int b = 0; b < 3; b++)
	 	for(int c = 0; c < 3; c++)
	 	for(int d = 0; d < 3; d++)
			Q2 += Q[a][b][c][d]*Q[a][b][c][d];	
	
	 
	 double norm = 1.2 * L3 * L3;

	 return Q2/norm;
}


double gauge_sim::nematicity_Ih()
{
	 double Q[3][3][3][3][3][3] = {{{{{{0}}}}}};
	 
	 double trace_Ih[3][3][3][3][3][3] = {{{{{{0}}}}}};

	 /* 2d delta, 3*3 to fit colors*/
	 Eigen::Matrix<double, 3, 3, Eigen::RowMajor> delta;
	 delta = Eigen::MatrixXd::Identity(3,3);

	 for(int a = 0; a < 3; a++)
	 	for(int b = 0; b < 3; b++)
	 	for(int c = 0; c < 3; c++)
	 	for(int d = 0; d < 3; d++)
	 	for(int e = 0; e < 3; e++)
	 	for(int f = 0; f < 3; f++)
	 		trace_Ih[a][b][c][d][e][f] 
	 		= (delta(a,b) * delta(c,d) * delta(e,f)
	 		   + delta(a,b) * delta(c,e) * delta(d,f)
	 		   + delta(a,b) * delta(c,f) * delta(d,e)
	 		   + delta(a,c) * delta(b,d) * delta(e,f)
	 		   + delta(a,c) * delta(b,e) * delta(d,f)
	 		   + delta(a,c) * delta(b,f) * delta(d,e)
	 		   + delta(a,d) * delta(b,c) * delta(e,f)
	 		   + delta(a,d) * delta(b,e) * delta(c,f)
	 		   + delta(a,d) * delta(b,f) * delta(c,e)
	 		   + delta(a,e) * delta(b,c) * delta(d,f)
	 		   + delta(a,e) * delta(b,d) * delta(c,f)
	 		   + delta(a,e) * delta(b,f) * delta(c,d)
	 		   + delta(a,f) * delta(b,c) * delta(d,e)
	 		   + delta(a,f) * delta(b,d) * delta(c,e)
	 		   + delta(a,f) * delta(b,e) * delta(c,d))/ 7.0;

/*
 (delta[a, b, c, d, e, f] +
  delta[a, b, c, e, d, f] + 
  delta[a, b, c, f, d, e] + 
  delta[a, c, b, d, e, f] + 
  delta[a, c, b, e, d, f] + 
  delta[a, c, b, f, d, e] + 
  delta[a, d, b, c, e, f] + 
  delta[a, d, b, e, c, f] + 
  delta[a, d, b, f, c, e] + 
  delta[a, e, b, c, d, f] + 
  delta[a, e, b, d, c, f] + 
  delta[a, e, b, f, c, d] + 
  delta[a, f, b, c, d, e] + 
  delta[a, f, b, d, c, e] + 
  delta[a, f, b, e, c, d])
*/
	/* 15 vectors for defining Ih order parameter*/ 		         
	double v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15;

	for(int i = 0; i < L3; i++)
		for(int a = 0; a < 3; a++)
		for(int b = 0; b < 3; b++)
		for(int c = 0; c < 3; c++)
		for(int d = 0; d < 3; d++)
		for(int e = 0; e < 3; e++)
		for(int f = 0; f < 3; f++)
		{
			v1 = R[i](0,a) * R[i](0,b) * R[i](0,c) * R[i](0,d) * R[i](0,e) * R[i](0,f);

			v2 = (R[i](0,a) + gr * R[i](1,a) + R[i](2,a) / gr) 
					* (R[i](0,b) + gr * R[i](1,b) + R[i](2,b) / gr)
					* (R[i](0,c) + gr * R[i](1,c) + R[i](2,c) / gr)
					* (R[i](0,d) + gr * R[i](1,d) + R[i](2,d) / gr)
					* (R[i](0,e) + gr * R[i](1,e) + R[i](2,e) / gr) 
					* (R[i](0,f) + gr * R[i](1,f) + R[i](2,f) / gr) / 64;

			v3 = (R[i](0,a) + gr * R[i](1,a) - R[i](2,a) / gr) 
					* (R[i](0,b) + gr * R[i](1,b) - R[i](2,b) / gr)
					* (R[i](0,c) + gr * R[i](1,c) - R[i](2,c) / gr)
					* (R[i](0,d) + gr * R[i](1,d) - R[i](2,d) / gr)
					* (R[i](0,e) + gr * R[i](1,e) - R[i](2,e) / gr) 
					* (R[i](0,f) + gr * R[i](1,f) - R[i](2,f) / gr) / 64;

			v4 = (R[i](0,a) - gr * R[i](1,a) + R[i](2,a) / gr) 
					* (R[i](0,b) - gr * R[i](1,b) + R[i](2,b) / gr)
					* (R[i](0,c) - gr * R[i](1,c) + R[i](2,c) / gr)
					* (R[i](0,d) - gr * R[i](1,d) + R[i](2,d) / gr)
					* (R[i](0,e) - gr * R[i](1,e) + R[i](2,e) / gr) 
					* (R[i](0,f) - gr * R[i](1,f) + R[i](2,f) / gr) / 64;

			v5 = (R[i](0,a) - gr * R[i](1,a) - R[i](2,a) / gr) 
					* (R[i](0,b) - gr * R[i](1,b) - R[i](2,b) / gr)
					* (R[i](0,c) - gr * R[i](1,c) - R[i](2,c) / gr)
					* (R[i](0,d) - gr * R[i](1,d) - R[i](2,d) / gr)
					* (R[i](0,e) - gr * R[i](1,e) - R[i](2,e) / gr) 
					* (R[i](0,f) - gr * R[i](1,f) - R[i](2,f) / gr) / 64;

			v6 = R[i](1,a) * R[i](1,b) * R[i](1,c) * R[i](1,d) * R[i](1,e) * R[i](1,f);

			v7 = (R[i](1,a) + gr * R[i](2,a) + R[i](0,a) / gr) 
					* (R[i](1,b) + gr * R[i](2,b) + R[i](0,b) / gr)
					* (R[i](1,c) + gr * R[i](2,c) + R[i](0,c) / gr)
					* (R[i](1,d) + gr * R[i](2,d) + R[i](0,d) / gr)
					* (R[i](1,e) + gr * R[i](2,e) + R[i](0,e) / gr) 
					* (R[i](1,f) + gr * R[i](2,f) + R[i](0,f) / gr) / 64;

			v8 = (R[i](1,a) + gr * R[i](2,a) - R[i](0,a) / gr) 
					* (R[i](1,b) + gr * R[i](2,b) - R[i](0,b) / gr)
					* (R[i](1,c) + gr * R[i](2,c) - R[i](0,c) / gr)
					* (R[i](1,d) + gr * R[i](2,d) - R[i](0,d) / gr)
					* (R[i](1,e) + gr * R[i](2,e) - R[i](0,e) / gr) 
					* (R[i](1,f) + gr * R[i](2,f) - R[i](0,f) / gr) / 64;

			v9 = (R[i](1,a) - gr * R[i](2,a) + R[i](0,a) / gr) 
					* (R[i](1,b) - gr * R[i](2,b) + R[i](0,b) / gr)
					* (R[i](1,c) - gr * R[i](2,c) + R[i](0,c) / gr)
					* (R[i](1,d) - gr * R[i](2,d) + R[i](0,d) / gr)
					* (R[i](1,e) - gr * R[i](2,e) + R[i](0,e) / gr) 
					* (R[i](1,f) - gr * R[i](2,f) + R[i](0,f) / gr) / 64;

			v10 = (R[i](1,a) - gr * R[i](2,a) - R[i](0,a) / gr) 
					* (R[i](1,b) - gr * R[i](2,b) - R[i](0,b) / gr)
					* (R[i](1,c) - gr * R[i](2,c) - R[i](0,c) / gr)
					* (R[i](1,d) - gr * R[i](2,d) - R[i](0,d) / gr)
					* (R[i](1,e) - gr * R[i](2,e) - R[i](0,e) / gr) 
					* (R[i](1,f) - gr * R[i](2,f) - R[i](0,f) / gr) / 64;

			v11 = R[i](2,a) * R[i](2,b) * R[i](2,c) * R[i](2,d) * R[i](2,e) * R[i](2,f);

			v12 = (R[i](2,a) + gr * R[i](0,a) + R[i](1,a) / gr) 
					* (R[i](2,b) + gr * R[i](0,b) + R[i](1,b) / gr)
					* (R[i](2,c) + gr * R[i](0,c) + R[i](1,c) / gr)
					* (R[i](2,d) + gr * R[i](0,d) + R[i](1,d) / gr)
					* (R[i](2,e) + gr * R[i](0,e) + R[i](1,e) / gr) 
					* (R[i](2,f) + gr * R[i](0,f) + R[i](1,f) / gr) / 64;

			v13 = (R[i](2,a) + gr * R[i](0,a) - R[i](1,a) / gr) 
					* (R[i](2,b) + gr * R[i](0,b) - R[i](1,b) / gr)
					* (R[i](2,c) + gr * R[i](0,c) - R[i](1,c) / gr)
					* (R[i](2,d) + gr * R[i](0,d) - R[i](1,d) / gr)
					* (R[i](2,e) + gr * R[i](0,e) - R[i](1,e) / gr) 
					* (R[i](2,f) + gr * R[i](0,f) - R[i](1,f) / gr) / 64;

			v14 = (R[i](2,a) - gr * R[i](0,a) + R[i](1,a) / gr) 
					* (R[i](2,b) - gr * R[i](0,b) + R[i](1,b) / gr)
					* (R[i](2,c) - gr * R[i](0,c) + R[i](1,c) / gr)
					* (R[i](2,d) - gr * R[i](0,d) + R[i](1,d) / gr)
					* (R[i](2,e) - gr * R[i](0,e) + R[i](1,e) / gr) 
					* (R[i](2,f) - gr * R[i](0,f) + R[i](1,f) / gr) / 64;

			v15 = (R[i](2,a) - gr * R[i](0,a) - R[i](1,a) / gr) 
					* (R[i](2,b) - gr * R[i](0,b) - R[i](1,b) / gr)
					* (R[i](2,c) - gr * R[i](0,c) - R[i](1,c) / gr)
					* (R[i](2,d) - gr * R[i](0,d) - R[i](1,d) / gr)
					* (R[i](2,e) - gr * R[i](0,e) - R[i](1,e) / gr) 
					* (R[i](2,f) - gr * R[i](0,f) - R[i](1,f) / gr) / 64;

			Q[a][b][c][d][e][f] += v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11+v12+v13+v14+v15 - trace_Ih[a][b][c][d][e][f];

		}	
	 				

	double Q2 = 0;

	 for(int a = 0; a < 3; a++)
	 	for(int b = 0; b < 3; b++)
	 	for(int c = 0; c < 3; c++)
	 	for(int d = 0; d < 3; d++)
	 	for(int e = 0; e < 3; e++)
	 	for(int f = 0; f < 3; f++)
			Q2 += Q[a][b][c][d][e][f] * Q[a][b][c][d][e][f];	
	
	 double norm = L3 * L3 * 75.0/112;

	 return Q2/norm;
}
