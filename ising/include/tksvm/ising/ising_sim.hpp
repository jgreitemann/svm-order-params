/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#include <tksvm/config/policy.hpp>
#include <tksvm/phase_space/classifier/policy.hpp>
#include <tksvm/phase_space/point/temperature.hpp>
#include <tksvm/sim_adapters/embarrassing_adapter.hpp>
#include <tksvm/phase_space/sweep/policy.hpp>

#include <tksvm/ising/config_policy.hpp>
#include <tksvm/ising/exp_beta.hpp>
#include <tksvm/ising/storage_type.hpp>


namespace tksvm {
namespace ising {

// Simulation class for 2D Ising model (square lattice).
// Extends alps::mcbase, the base class of all Monte Carlo simulations.
// Defines its state, calculation functions (update/measure) and
// serialization functions (save/load)
class ising_sim : public embarrassing_adapter<phase_space::point::temperature> {
public:
    using phase_point = phase_space::point::temperature;
    using Base = embarrassing_adapter<phase_point>;
    using phase_label = typename phase_space::classifier::policy<phase_point>::label_type;
    using phase_sweep_policy_type = phase_space::sweep::policy<phase_point>;
private:
    int length; // the same in both dimensions
    int sweeps;
    int thermalization_sweeps;
    int total_sweeps;
    phase_point ppoint;
    storage_type spins;
    double current_energy;
    double current_magnetization;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform;
    std::uniform_int_distribution<size_t> random_site;

    exp_beta iexp_; // function object to compute exponent

public:
    ising_sim(parameters_type & parms, std::size_t seed_offset = 0);

    static void define_parameters(parameters_type & parameters);

    // SVM interface functions
    std::vector<std::string> order_param_names() const {
        return {"Magnetization^2"};
    }
    virtual void reset_sweeps(bool skip_therm = false) override;
    bool is_thermalized() const;
    storage_type const& configuration() const;
    storage_type random_configuration();
    virtual phase_point phase_space_point () const override;
    virtual bool update_phase_point(phase_point const&) override;

    template <typename Introspector>
    using config_policy_type = config::policy<storage_type, Introspector>;

    template <typename Introspector>
    static auto config_policy_from_parameters(parameters_type const& parameters,
                                              bool unsymmetrize = true)
        -> std::unique_ptr<config_policy_type<Introspector>>
    {
        return ising::config_policy_from_parameters<storage_type, Introspector>(
            parameters, unsymmetrize);
    }

    virtual void update() override;
    virtual void measure() override;
    virtual double fraction_completed() const override;

    using Base::save;
    using Base::load;
    virtual void save(alps::hdf5::archive & ar) const override;
    virtual void load(alps::hdf5::archive & ar) override;
};

}

using sim_base = ising::ising_sim;

}
