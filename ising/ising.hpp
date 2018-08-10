/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include "storage_type.hpp"
#include "exp_beta.hpp"
#include "phase_space_policy.hpp"

#include <random>

#include <alps/mc/mcbase.hpp>


// Simulation class for 2D Ising model (square lattice).
// Extends alps::mcbase, the base class of all Monte Carlo simulations.
// Defines its state, calculation functions (update/measure) and
// serialization functions (save/load)
class ising_sim : public alps::mcbase {
public:
    using phase_classifier = phase_space::classifier::critical_temperature;
    using phase_label = phase_classifier::label_type;
    using phase_point = phase_classifier::point_type;
    using phase_sweep_policy_type = phase_space::sweep::policy<phase_point>;
private:
    int length; // the same in both dimensions
    int sweeps;
    int thermalization_sweeps;
    int total_sweeps;
    phase_point ppoint;
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
    std::vector<std::string> order_param_names() const {
        return {"Magnetization^2"};
    }
    void reset_sweeps(bool skip_therm = false);
    bool is_thermalized() const;
    size_t configuration_size() const;
    std::vector<int> const& configuration() const;
    phase_point phase_space_point () const;
    void update_phase_point (phase_sweep_policy_type & sweep_policy);

    virtual void update();
    virtual void measure();
    virtual double fraction_completed() const;

    using alps::mcbase::save;
    using alps::mcbase::load;
    virtual void save(alps::hdf5::archive & ar) const;
    virtual void load(alps::hdf5::archive & ar);
};
