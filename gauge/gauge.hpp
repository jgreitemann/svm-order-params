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

#include "config_serialization.hpp"
#include "embarrassing_adapter.hpp"
#include "point_groups.hpp"
#include "phase_space_policy.hpp"
#include "gauge_config_policy.hpp"

#include <cmath>
#include <memory>
#include <random>
#include <string>

#include <alps/numeric/vector_functions.hpp>

#include <boost/math/constants/constants.hpp>
#include <boost/multi_array.hpp>

#include <Eigen/Dense>


constexpr double pi2(boost::math::constants::two_pi<double>());

class gauge_sim : public embarrassing_adapter<phase_space::point::J1J3> {
public:
    using Base = embarrassing_adapter<phase_space::point::J1J3>;
    using phase_point = phase_space::point::J1J3;
#if defined(GAUGE_CLASSIFIER_HYPERPLANE)
    using phase_classifier = phase_space::classifier::hyperplane<phase_point>;
#elif defined(GAUGE_CLASSIFIER_CYCLE)
    using phase_classifier = phase_space::classifier::fixed_from_cycle<phase_point>;
#elif defined(GAUGE_CLASSIFIER_D2H)
    using phase_classifier = phase_space::classifier::D2h_phase_diagram;
#elif defined(GAUGE_CLASSIFIER_D3H)
    using phase_classifier = phase_space::classifier::D3h_phase_diagram;
#elif defined(GAUGE_CLASSIFIER_GRID)
    using phase_classifier = phase_space::classifier::fixed_from_grid<phase_point>;
#elif defined(GAUGE_CLASSIFIER_NONUNIFORM_GRID)
    using phase_classifier = phase_space::classifier::fixed_from_nonuniform_grid<phase_point>;
#else
    #error unknown / missing gauge classifier
#endif
    using phase_label = phase_classifier::label_type;
    using phase_sweep_policy_type = phase_space::sweep::policy<phase_point>;
private:
    int L;
    int L2;
    int L3;

    /* MC parameters */
    int sweeps = 0;
    int thermalization_sweeps;
    int total_sweeps;
    int sweep_unit;
    int hits_R; // number of hits in each update of R
    int hits_U;
    double global_gauge_prob;

    const bool O3; // promote SO(3) to O(3) or not
    std::string gauge_group;
    int group_size;

    phase_point ppoint;
    double beta;
    double J1, J3; // define anisotropy of J matrix

    /* for histogram */
    double spacing_E;
    double spacing_nem;

public:
    /**  degrees of freedoms **/
    typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Rt3;
    typedef boost::multi_array<Rt3, 1> Rt_array;
private:

    Rt_array R; //matter fields

    Rt_array gauge_bath; // store group elements
    Rt_array Ux; // gauge fields along x direction
    Rt_array Uy;
    Rt_array Uz;

    Rt3 J;

    /* nematicity callbacks */
    point_groups pg;
    std::function<double()> nematicity;
    std::function<double()> nematicityB;  //biaxial nematicity
    std::function<double()> nematicityB2; // variant biaxial nematicity

    /** measurements **/
    double current_energy;
    double Eg; // ground state energy

    /** random number generators**/
    std::mt19937 rng;

    std::uniform_real_distribution<double> random_01{0, 1};
    std::uniform_real_distribution<double> random_11{-1, 1};
    std::uniform_int_distribution<int> random_int_01{0, 1};
    std::uniform_int_distribution<int> random_site;
    std::uniform_int_distribution<int> random_U;


    /* update counter; the number of succeeded updates for R,
       Ux, Uy, Uz, respectively */
    std::vector<double> flip_counter = { 0, 0, 0, 0 };

public:
    gauge_sim(parameters_type & parms, std::size_t seed_offset = 0);
    virtual ~gauge_sim();

    static void define_parameters(parameters_type & parameters);

    template <typename Introspector>
    using config_policy_type = config_policy<Rt_array, Introspector>;

    template <typename Introspector>
    static auto config_policy_from_parameters(parameters_type const& parameters,
                                              bool unsymmetrize = true)
        -> std::unique_ptr<config_policy_type<Introspector>>
    {
        return gauge_config_policy_from_parameters<Rt_array, Introspector>(
            parameters, unsymmetrize);
    }

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

    using Base::save;
    using Base::load;
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
    Rt_array const& configuration() const {
        return R;
    }
    Rt_array random_configuration();
    virtual void reset_sweeps(bool skip_therm) override;
    bool is_thermalized() const;
    virtual phase_point phase_space_point() const override;
    virtual bool update_phase_point(phase_point const&) override;
};

template <>
struct config_serializer<gauge_sim::Rt_array> {
    using config_t = gauge_sim::Rt_array;
    template <typename OutputIterator>
    void serialize(config_t const& conf, OutputIterator it) const {
        for (auto const& mat : conf) {
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    *(it++) = mat(i, j);
                }
            }
        }
    }

    template <typename InputIterator>
    void deserialize(InputIterator it, config_t & conf) const {
        for (auto & mat : conf) {
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    mat(i, j) = *(it++);
                }
            }
        }
    }
};