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

#include "concepts.hpp"
#include "convenience_params.hpp"
#include "observables.hpp"
#include "phase_space_point.hpp"
#ifdef HAS_SVM
#include "frustmag_config_policy.hpp"
#include "phase_space_policy.hpp"
#endif

#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>

#include <algorithm>
#include <cstdlib>
#include <random>
#include <sstream>
#include <vector>

template <typename Hamiltonian, template <typename> typename Update>
class frustmag_sim : public alps::mcbase, Update<Hamiltonian> {
    using rng_type = std::mt19937;
#ifdef USE_CONCEPTS
    static_assert(LatticeHamiltonian<Hamiltonian, rng_type>,
                  "Hamiltonian is not a LatticeHamiltonian");
    static_assert(MagneticHamiltonian<Hamiltonian>,
                  "Hamiltonian is not a MagneticHamiltonian");
    static_assert(MCUpdate<Update<Hamiltonian>, rng_type>,
                  "Update is not a MCUpdate");
#endif
public:
    using phase_point = phase_space::point::temperature;
    using lattice_type = typename Hamiltonian::lattice_type;
    using update_type = Update<Hamiltonian>;

#ifdef HAS_SVM
    using phase_classifier = phase_space::classifier::critical_temperature;
    using phase_label = phase_classifier::label_type;
    using phase_sweep_policy_type = phase_space::sweep::policy<phase_point>;
    using test_sweep_type = phase_space::sweep::line_scan<phase_point>;

    template <typename Introspector>
    using config_policy_type = config_policy<lattice_type, Introspector>;

    template <typename Introspector>
    static auto config_policy_from_parameters(parameters_type const& parameters,
                                              bool unsymmetrize = true)
        -> std::unique_ptr<config_policy_type<Introspector>>
    {
        return frustmag_config_policy_from_parameters<lattice_type, Introspector>(
            parameters, unsymmetrize);
    }
#endif
private:
    size_t sweeps = 0;
    size_t total_sweeps;
    size_t thermalization_sweeps;

    rng_type rng;

    Hamiltonian hamiltonian_;

    std::vector<double> acceptance;

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
            .description("Simulation of frustrated magnetism")
            .define<size_t>("total_sweeps", 0,
                            "maximum number of sweeps (0 means indefinite)")
            .define<size_t>("thermalization_sweeps", 10000,
                            "number of sweeps for thermalization");

        update_type::define_parameters(parameters);

        Hamiltonian::define_parameters(parameters);

#ifdef HAS_SVM
        define_frustmag_config_policy_parameters(parameters);
#endif
    }

    frustmag_sim(parameters_type const & params, std::size_t seed_offset = 0)
        : alps::mcbase{params, seed_offset}
        , update_type{params}
        , total_sweeps{params["total_sweeps"]}
        , thermalization_sweeps{params["thermalization_sweeps"]}
        , rng{params["SEED"].as<size_t>() + seed_offset}
        , hamiltonian_{params, rng}
    {
        observables<Hamiltonian>::define(measurements);
        measurements
            << alps::accumulators::FullBinningAccumulator<std::vector<double>>("Acceptance");

        using acc_t = typename update_type::acceptance_type;
        acceptance.resize(acc_t{}.size());
    }

    Hamiltonian const& hamiltonian() const {
        return hamiltonian_;
    }

    virtual void update() {
        auto acc = update_type::update(hamiltonian_, rng);
        std::copy(acc.begin(), acc.end(), acceptance.begin());
        measurements["Acceptance"] << acceptance;
    }

    virtual void measure() {
        ++sweeps;
        if (!is_thermalized()) return;

        measurements << observables<Hamiltonian>{hamiltonian_};
    }

    virtual double fraction_completed() const {
        if (total_sweeps > 0 && is_thermalized()) {
            return (sweeps - thermalization_sweeps) / double(total_sweeps);
        }
        return 0;

    }

    using alps::mcbase::save;
    using alps::mcbase::load;
    virtual void save(alps::hdf5::archive & ar) const {
        alps::mcbase::save(ar);

        {
            std::stringstream engine_ss;
            engine_ss << rng;
            ar["checkpoint/random"] << engine_ss.str();
        }

        ar["checkpoint/sweeps"] << sweeps;
        ar["checkpoint/hamiltonian"] << hamiltonian_;
    }
    virtual void load(alps::hdf5::archive & ar) {
        alps::mcbase::load(ar);

        {
            std::string engine_str;
            ar["checkpoint/random"] >> engine_str;
            std::istringstream engine_ss(engine_str);
            engine_ss >> rng;
        }

        ar["checkpoint/sweeps"] >> sweeps;
        ar["checkpoint/hamiltonian"] >> hamiltonian_;
    }

    // SVM interface functions
    std::vector<std::string> order_param_names() const {
        return observables<Hamiltonian>::names();
    }

    void reset_sweeps(bool skip_therm) {
        if (skip_therm)
            sweeps = thermalization_sweeps;
        else
            sweeps = 0;
    }

    bool is_thermalized() const {
        return sweeps > thermalization_sweeps;
    }

    lattice_type const& configuration() const {
        return hamiltonian_.lattice();
    }

    lattice_type random_configuration() {
        lattice_type random_lattice = hamiltonian_.lattice();
        using site_t = typename lattice_type::value_type;
        std::generate(random_lattice.begin(), random_lattice.end(),
            [&] { return site_t::random(rng); });
        return random_lattice;
    }

    phase_point phase_space_point() const {
        return hamiltonian_.phase_space_point();
    }

    bool update_phase_point(phase_point const& pp) {
        auto const& current_pp = hamiltonian_.phase_space_point();
        bool changed = (current_pp != pp);
        if (changed)
            hamiltonian_.phase_space_point(pp);
        return changed;
    }
};
