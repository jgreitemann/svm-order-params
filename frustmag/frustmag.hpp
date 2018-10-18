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

#include "convenience_params.hpp"

#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>

#include <cstdlib>
#include <random>

template <typename Hamiltonian, template <typename> typename Update>
class frustmag_sim : public alps::mcbase, Update<Hamiltonian> {
public:
    // using phase_classifier = phase_space::classifier::critical_temperature;
    // using phase_label = phase_classifier::label_type;
    // using phase_point = phase_classifier::point_type;
    // using phase_sweep_policy_type = phase_space::sweep::policy<phase_point>;

    using lattice_type = typename Hamiltonian::lattice_type;
    using update_type = Update<Hamiltonian>;
    struct state_type {
    };
private:
    size_t sweeps = 0;
    size_t total_sweeps;
    size_t thermalization_sweeps;

    std::mt19937 rng;

    Hamiltonian hamiltonian;

public:
    static void define_parameters(parameters_type & parameters) {
        // If the parameters are restored, they are already defined
        if (parameters.is_restored()) {
            return;
        }
    
        // Adds the parameters of the base class
        alps::mcbase::define_parameters(parameters);
        // Adds the convenience parameters (for save/load)
        // followed by simulation control parameters
        define_convenience_parameters(parameters)
            .description("Simulation of the TODO")
            .define<size_t>("total_sweeps", 0,
                            "maximum number of sweeps (0 means indefinite)")
            .define<size_t>("thermalization_sweeps", 10000,
                            "number of sweeps for thermalization");

        Hamiltonian::define_parameters(parameters);
    }

    frustmag_sim(parameters_type const & params, std::size_t seed_offset = 0)
        : alps::mcbase{params, seed_offset}
        , update_type{}
        , total_sweeps{params["total_sweeps"]}
        , thermalization_sweeps{params["thermalization_sweeps"]}
        , rng{params["SEED"].as<size_t>() + seed_offset}
        , hamiltonian{params, rng}
    {
        measurements
            << alps::accumulators::FullBinningAccumulator<double>("Energy");
    }


    bool is_thermalized() const {
        return sweeps > thermalization_sweeps;
    }

    virtual void update() {
        update_type::update(hamiltonian, rng);
    }

    virtual void measure() {
        ++sweeps;
        if (!is_thermalized()) return;

        measurements["Energy"] << hamiltonian.energy_per_site();
    }

    virtual double fraction_completed() const {
        if (total_sweeps > 0 && is_thermalized()) {
            return (sweeps - thermalization_sweeps) / double(total_sweeps);
        }
        return 0;
        
    }

    using alps::mcbase::save;
    using alps::mcbase::load;
    virtual void save(alps::hdf5::archive & ar) const {}
    virtual void load(alps::hdf5::archive & ar) {}
};
