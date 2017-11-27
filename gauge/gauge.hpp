#pragma once

#include <alps/mc/mcbase.hpp>
#include <time.h>
#include <random>
#include <cmath>
#include <string>
#include <boost/math/constants/constants.hpp>
#include <boost/multi_array.hpp>
#include <boost/function.hpp>
#include <alps/numeric/vector_functions.hpp>
#include "Eigen/Dense"
#include "point_groups.hpp"

#include <iterator>
#include "binomial.hpp"


constexpr double pi2(boost::math::constants::two_pi<double>());


class gauge_sim : public alps::mcbase {
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
    int sweep_unit; // the actual unit is sweep_unit*hits

    /* for histogram */
    double spacing_E;
    double spacing_nem;

    double beta; // used in each updates, modified in warm_up()

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
    boost::function<double ()> nematicity;
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

    static void define_parameters(parameters_type & parameters);

    virtual void update();
    virtual void measure();
    virtual double fraction_completed() const;


    /** contents of update **/
    void random_R(Rt3& Rt); // re-set the value of R

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

    /* compute succeeded flip ratio */
    void flip_ratio(double current_beta, int N);

    /* Histogram */
    void histogram();

    using alps::mcbase::save;
    using alps::mcbase::load;
    virtual void save(alps::hdf5::archive & ar) const;
    virtual void load(alps::hdf5::archive & ar);

    // SVM interface functions
    static constexpr const char * order_param_name = "Nematicity";
    void reset_sweeps(bool skip_therm);
    void temperature(double new_temp);
    bool is_thermalized() const;

    template <size_t Rank = 2, size_t Range = 3>
    size_t configuration_size() const {
        return binomial(Rank + Range - 1, Rank);
    }

    template <size_t Rank = 2, size_t Range = 3>
    std::vector<double> configuration() const {
        std::vector<double> v(configuration_size<Rank,Range>());

        std::array<size_t, Rank> ind = {};
        for (double & elem : v) {
            for (Rt3 const& site : R) {
                double prod = 1;
                for (size_t a : ind)
                    prod *= site(2, a);
                elem += prod;
            }
            elem /= L3;

            auto it = ind.begin();
            ++(*it);
            while (*it == Range) {
                ++it;
                if (it == ind.end())
                    break;
                ++(*it);
                std::reverse_iterator<decltype(it)> rit(it);
                while (rit != ind.rend()) {
                    *rit = *it;
                    ++rit;
                }
            }
        }
        return v;
    }

};
