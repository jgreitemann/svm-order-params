/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "ising.hpp"
#include "convenience_params.hpp"

#include <algorithm>
#include <sstream>


// Defines the parameters for the ising simulation
void ising_sim::define_parameters(parameters_type & parameters) {
    // If the parameters are restored, they are already defined
    if (parameters.is_restored()) {
        return;
    }

    // Adds the parameters of the base class
    alps::mcbase::define_parameters(parameters);
    // Adds the convenience parameters (for save/load)
    // followed by the ising specific parameters
    define_convenience_parameters(parameters)
        .description("2D ising simulation")
        .define<int>("length", "size of the periodic box")
        .define<int>("total_sweeps", 0, "maximum number of sweeps (0 means indefinite)")
        .define<int>("thermalization_sweeps", 10000, "number of sweeps for thermalization")
        .define<double>("temperature", "temperature of the system");
    phase_point::define_parameters(parameters);

    define_ising_config_policy_parameters(parameters);
}

// Creates a new simulation.
// We always need the parameters and the seed as we need to pass it to
// the alps::mcbase constructor. We also initialize our internal state,
// mainly using values from the parameters.
ising_sim::ising_sim(parameters_type const & parms, std::size_t seed_offset)
    : alps::mcbase(parms, seed_offset)
    , length(parameters["length"])
    , sweeps(0)
    , thermalization_sweeps(int(parameters["thermalization_sweeps"]))
    , total_sweeps(parameters["total_sweeps"])
    , ppoint(parameters["temperature"].as<double>())
    , spins(length,length)
    , current_energy(0)
    , current_magnetization(0)
    , rng(parameters["SEED"].as<size_t>() + seed_offset)
    , uniform(0., 1.)
    , random_site(0, length - 1)
    , iexp_(-1. / ppoint.temp)
{
    // Initializes the spins
    for(int i=0; i<length; ++i) {
        for (int j=0; j<length; ++j) {
            spins(i,j) = (random() < 0.5 ? 1 : -1);
        }
    }

    // Calculates initial magnetization and energy
    for (int i=0; i<length; ++i) {
        for (int j=0; j<length; ++j) {
            current_magnetization += spins(i,j);
            int i_next=(i+1)%length;
            int j_next=(j+1)%length;
            current_energy += -(spins(i,j)*spins(i,j_next)+
                                spins(i,j)*spins(i_next,j));

        }
    }

    // Adds the measurements
    measurements
        << alps::accumulators::FullBinningAccumulator<double>("Energy")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization")
        << alps::accumulators::FullBinningAccumulator<double>("AbsMagnetization")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization^2")
        << alps::accumulators::FullBinningAccumulator<double>("Magnetization^4")
        ;
}

void ising_sim::reset_sweeps(bool skip_therm) {
    if (skip_therm)
        sweeps = thermalization_sweeps;
    else
        sweeps = 0;
}

bool ising_sim::is_thermalized() const {
    return sweeps > thermalization_sweeps;
}

storage_type const& ising_sim::configuration() const {
    return spins;
}

storage_type ising_sim::random_configuration() {
    auto random_spins = spins;
    std::generate(random_spins.begin(), random_spins.end(),
        [&]() {
            return std::bernoulli_distribution{}(rng) ? 1. : -1;
        });
    return random_spins;
}

ising_sim::phase_point ising_sim::phase_space_point () const {
    return ppoint;
}

void ising_sim::update_phase_point(phase_point pp) {
    bool changed = (pp != ppoint);
    reset_sweeps(!changed);
    if (changed) {
        ppoint = pp;
        parameters["temperature"] = ppoint.temp;
        iexp_ = exp_beta(-1. / ppoint.temp);
    }
}

// Performs the calculation at each MC step;
// decides if the step is accepted.
void ising_sim::update() {
    using std::exp;
    // Choose a spin to flip:
    size_t i = random_site(rng);
    size_t j = random_site(rng);
    // Find neighbors indices, with wrap over box boundaries:
    size_t i1 = (i+1) % length;            // right
    size_t i2 = (i-1+length) % length;     // left
    size_t j1 = (j+1) % length;            // up
    size_t j2 = (j-1+length) % length;     // down
    // Energy difference:
    double delta=2.*spins(i,j)*
                    (spins(i1,j)+  // right
                     spins(i2,j)+  // left
                     spins(i,j1)+  // up
                     spins(i,j2)); // down

    // Step acceptance:
    if (delta<=0. || uniform(rng) < iexp_(delta)) {
        // update energy:
        current_energy += delta;
        // update magnetization:
        current_magnetization -= 2*spins(i,j);
        // flip the spin
        spins(i,j) = -spins(i,j);
    }
}

// Collects the measurements at each MC step.
void ising_sim::measure() {
    ++sweeps;
    if (!is_thermalized()) return;

    const double n=length*length; // number of sites
    double tmag = current_magnetization / n; // magnetization

    // Accumulate the data (per site)
    measurements["Energy"] << (current_energy / n);
    measurements["Magnetization"] << tmag;
    measurements["AbsMagnetization"] << fabs(tmag);
    measurements["Magnetization^2"] << tmag*tmag;
    measurements["Magnetization^4"] << tmag*tmag*tmag*tmag;
}

// Returns a number between 0.0 and 1.0 with the completion percentage
double ising_sim::fraction_completed() const {
    double f=0;
    if (total_sweeps > 0 && is_thermalized()) {
        f=(sweeps-thermalization_sweeps)/double(total_sweeps);
    }
    return f;
}

// Saves the state to the hdf5 file
void ising_sim::save(alps::hdf5::archive & ar) const {
    // Most of the save logic is already implemented in the base class
    alps::mcbase::save(ar);

    // random number engine
    std::ostringstream engine_ss;
    engine_ss << rng;
    ar["checkpoint/random"] << engine_ss.str();

    // We just need to add our own internal state
    ar["checkpoint/spins"] << spins;
    ar["checkpoint/sweeps"] << sweeps;
    ar["checkpoint/current_energy"] << current_energy;
    ar["checkpoint/current_magnetization"] << current_magnetization;

    // The rest of the internal state is saved as part of the parameters
}

// Loads the state from the hdf5 file
void ising_sim::load(alps::hdf5::archive & ar) {
    // Most of the load logic is already implemented in the base class
    alps::mcbase::load(ar);

    // random number engine
    std::string engine_str;
    ar["checkpoint/random"] >> engine_str;
    std::istringstream engine_ss(engine_str);
    engine_ss >> rng;

    // Restore the internal state that came from parameters
    length = parameters["length"];
    thermalization_sweeps = parameters["thermalization"];
    // Note: `total_sweeps` is not restored here!
    ppoint.temp = parameters["temperature"].as<double>();
    iexp_ = exp_beta(-1. / ppoint.temp);

    // Restore the rest of the state from the hdf5 file
    ar["checkpoint/spins"] >> spins;
    ar["checkpoint/sweeps"] >> sweeps;
    ar["checkpoint/current_energy"] >> current_energy;
    ar["checkpoint/current_magnetization"] >> current_magnetization;
}
