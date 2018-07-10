/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include "storage_type.hpp"
#include "exp_beta.hpp"

#include "label.hpp"
#include <random>

#include <alps/mc/mcbase.hpp>

SVM_LABEL_BEGIN(ising_phase_label, 2)
SVM_LABEL_ADD(ORDERED)
SVM_LABEL_ADD(Z2)
SVM_LABEL_END()

// Simulation class for 2D Ising model (square lattice).
// Extends alps::mcbase, the base class of all Monte Carlo simulations.
// Defines its state, calculation functions (update/measure) and
// serialization functions (save/load)
class ising_sim : public alps::mcbase {
public:
    typedef ising_phase_label::label phase_label;

    struct phase_point {
        static const size_t label_dim = 1;
        phase_point(double temp) : temp(temp) {}
        template <class Iterator>
        phase_point(Iterator begin) : temp(*begin) {}
        double const * begin() const { return &temp; }
        double const * end() const { return &temp + 1; }

        double const temp;
    };

    struct phase_classifier {
        phase_classifier(alps::params const& params);
        phase_label operator() (phase_point pp);
    private:
        double temp_crit;
    };

private:
    int length; // the same in both dimensions
    int sweeps;
    int thermalization_sweeps;
    int total_sweeps;
    double beta;
    storage_type spins;
    double current_energy;
    double current_magnetization;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform;
    std::uniform_int_distribution<size_t> random_site;

    exp_beta iexp_; // function object to compute exponent

  public:
    ising_sim(parameters_type const & parms, std::size_t seed_offset = 0);

    static void define_parameters(parameters_type & parameters);

    // SVM interface functions
    static constexpr const char * order_param_name = "Magnetization^2";
    void reset_sweeps(bool skip_therm = false);
    void temperature(double new_temp);
    bool is_thermalized() const;
    size_t configuration_size() const;
    std::vector<int> const& configuration() const;

    virtual void update();
    virtual void measure();
    virtual double fraction_completed() const;

    using alps::mcbase::save;
    using alps::mcbase::load;
    virtual void save(alps::hdf5::archive & ar) const;
    virtual void load(alps::hdf5::archive & ar);
};
