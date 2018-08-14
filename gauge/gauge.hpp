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

#pragma once

#include "point_groups.hpp"
#include "phase_space_policy.hpp"

#include <cmath>
#include <memory>
#include <random>
#include <string>

#include <alps/numeric/vector_functions.hpp>
#include <alps/mc/mcbase.hpp>

#include <boost/math/constants/constants.hpp>
#include <boost/multi_array.hpp>

#include <Eigen/Dense>


constexpr double pi2(boost::math::constants::two_pi<double>());

// forward declaration
struct config_policy;

class gauge_sim : public alps::mcbase {
public:
    using phase_point = phase_space::point::J1J3;
    using phase_classifier = phase_space::classifier::hyperplane<phase_point>;
    // using phase_classifier = phase_space::classifier::fixed_from_cycle<phase_point, 4>;
    using phase_label = phase_classifier::label_type;
    using phase_sweep_policy_type = phase_space::sweep::policy<phase_point>;
private:
    /** parameters **/
    int L;
    int L2;
    int L3;

    int sweeps;
    int thermalization_sweeps;
    int total_sweeps;
    int hits_R; // number of hits in each update of R
    int hits_U;
    double global_gauge_prob;
    int sweep_unit; // the actual unit is sweep_unit*hits

    /* for histogram */
    double spacing_E;
    double spacing_nem;

    phase_point ppoint;
    double beta;

    double J1, J3; // define anisotropy of J matrix

    double Eg; // ground state energy

    /**  degrees of freedoms **/
    typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Rt3;
    typedef boost::multi_array<Rt3, 1> Rt_array;

    Rt_array R; //matter fields

    Rt_array gauge_bath; // store group elements
    Rt_array Ux; // gauge fields along x direction
    Rt_array Uy;
    Rt_array Uz;

    Rt3 J;


    /* promot SO(3) to O(3) or not */
    const bool O3;

    std::string gauge_group;
    std::function<double ()> nematicity;
    std::function<double ()> nematicityB;  //biaxial nematicity
    std::function<double ()> nematicityB2; // variant biaxial nematicity
    int group_size;
    point_groups pg; // defined in point_groups.hpp

    /** measurements **/
    double current_energy;

    /** random number generators**/
    std::mt19937 rng;

    std::uniform_real_distribution<double> random_01;
    std::uniform_real_distribution<double> random_11;
    std::uniform_int_distribution<int> random_site;
    std::uniform_int_distribution<int> random_U;
    std::uniform_int_distribution<int> random_int_01;


    /* update counter; the number of succeeded updates for R,
       Ux, Uy, Uz, respectively */
    std::vector<double> flip_counter;

public:
    gauge_sim(parameters_type const & parms, std::size_t seed_offset = 0);
    virtual ~gauge_sim();

    static void define_parameters(parameters_type & parameters);

    static std::unique_ptr<config_policy> config_policy_from_parameters(parameters_type const&, bool);

    virtual void update();
    virtual void measure();
    virtual double fraction_completed() const;


    /** contents of update **/
    void random_R(Rt3& Rt); // re-set the value of R

    void global_gauge_update();
    void flip_R(int i);
    void flip_Ux(int i);
    void flip_Uy(int i);
    void flip_Uz(int i);

    /** measurements **/
    double total_energy();

    /* nematicities */
    double nematicity_Cinfv();
    double nematicity_Dinfh();
    double nematicity_T();
    double nematicity_Td();
    double nematicity_Th();
    double nematicity_Oh();
    double nematicity_Ih();

    double nematicity_D2hB();
    double nematicity_D2hB2();
    double nematicity_D2dB();
    double nematicity_D3hB();

    /* compute succeeded flip ratio */
    void flip_ratio(double current_beta, int N);

    /* Histogram */
    void histogram();

    using alps::mcbase::save;
    using alps::mcbase::load;
    virtual void save(alps::hdf5::archive & ar) const;
    virtual void load(alps::hdf5::archive & ar);

    // SVM interface functions
    std::vector<std::string> order_param_names() const {
        std::vector<std::string> names = {"Nematicity"};
        if (nematicityB)
            names.push_back("NematicityB");
        if (nematicityB2)
            names.push_back("NematicityB2");
        return names;
    }
    void reset_sweeps(bool skip_therm);
    bool is_thermalized() const;
    size_t configuration_size() const;
    std::vector<double> configuration() const;
    phase_point phase_space_point () const;
    void update_phase_point (phase_sweep_policy_type & sweep_policy);

private:
    std::unique_ptr<config_policy> confpol;
};
