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

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <tksvm/config/policy.hpp>
#include <tksvm/phase_space/classifier/policy.hpp>
#include <tksvm/phase_space/sweep/policy.hpp>
#include <tksvm/sim_adapters/pt_adapter.hpp>
#include <tksvm/utilities/convenience_params.hpp>

#include <tksvm/frustmag/concepts.hpp>
#include <tksvm/frustmag/config_policy.hpp>
#include <tksvm/frustmag/observables.hpp>
#include <tksvm/frustmag/phase_diagram.hpp>


namespace tksvm {
namespace frustmag {

template <typename Hamiltonian, template <typename> typename Update>
struct frustmag_sim : public pt_adapter<typename Hamiltonian::phase_point>
                    , protected Update<Hamiltonian>
{
    using Base = pt_adapter<typename Hamiltonian::phase_point>;
    using parameters_type = typename Base::parameters_type;
    using rng_type = std::mt19937;
#ifdef USE_CONCEPTS
    static_assert(LatticeHamiltonian<Hamiltonian, rng_type>,
                  "Hamiltonian is not a LatticeHamiltonian");
    static_assert(MagneticHamiltonian<Hamiltonian>,
                  "Hamiltonian is not a MagneticHamiltonian");
    static_assert(MCUpdate<Update<Hamiltonian>, rng_type>,
                  "Update is not a MCUpdate");
#endif
    using phase_point = typename Hamiltonian::phase_point;
    using lattice_type = typename Hamiltonian::lattice_type;
    using update_type = Update<Hamiltonian>;

    template <typename Introspector>
    using config_policy_type = config::policy<lattice_type, Introspector>;

    template <typename Introspector>
    static auto config_policy_from_parameters(parameters_type const& parameters,
                                              bool unsymmetrize = true)
        -> std::unique_ptr<config_policy_type<Introspector>>
    {
        return frustmag::config_policy_from_parameters<lattice_type, Introspector>(
            parameters, unsymmetrize);
    }
protected:
    using Base::measurements;
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
        Base::define_parameters(parameters);
        // Adds the convenience parameters (for save/load)
        // followed by simulation control parameters
        define_convenience_parameters(parameters)
            .description("Simulation of frustrated magnetism")
            .template define<size_t>("total_sweeps", 0,
                "maximum number of sweeps (0 means indefinite)")
            .template define<size_t>("thermalization_sweeps", 10000,
                "number of sweeps for thermalization");

        update_type::define_parameters(parameters);

        Hamiltonian::define_parameters(parameters);

        define_config_policy_parameters(parameters);
    }

    frustmag_sim(parameters_type & params, std::size_t seed_offset = 0)
        : Base{params, seed_offset}
        , update_type{params, *this}
        , total_sweeps{params["total_sweeps"]}
        , thermalization_sweeps{params["thermalization_sweeps"]}
        , rng{params["SEED"].template as<size_t>() + seed_offset}
        , hamiltonian_{params, rng}
    {
        observables<Hamiltonian>::define(measurements());
        measurements()
            << alps::accumulators::FullBinningAccumulator<std::vector<double>>("Acceptance");
        measurements()
            << alps::accumulators::FullBinningAccumulator<std::vector<double>>("Point");

        using acc_t = typename update_type::acceptance_type;
        acceptance.resize(acc_t{}.size());
    }

    Hamiltonian const& hamiltonian() const {
        return hamiltonian_;
    }

    virtual void update() override {
        auto acc = update_type::update(hamiltonian_, rng);
        std::copy(acc.begin(), acc.end(), acceptance.begin());
        measurements()["Acceptance"] << acceptance;
    }

    virtual void measure() override {
        ++sweeps;
        if (!is_thermalized()) return;

        measurements() << observables<Hamiltonian>{hamiltonian_};
        auto pt = phase_space_point();
        measurements()["Point"] << std::vector<double>{pt.begin(), pt.end()};
    }

    virtual double fraction_completed() const override {
        if (total_sweeps > 0 && is_thermalized()) {
            return (sweeps - thermalization_sweeps) / double(total_sweeps);
        }
        return 0;

    }

    using Base::save;
    using Base::load;
    virtual void save(alps::hdf5::archive & ar) const override {
        Base::save(ar);

        {
            std::stringstream engine_ss;
            engine_ss << rng;
            ar["checkpoint/random"] << engine_ss.str();
        }

        ar["checkpoint/sweeps"] << sweeps;
        ar["checkpoint/hamiltonian"] << hamiltonian_;
    }
    virtual void load(alps::hdf5::archive & ar) override {
        Base::load(ar);

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

    virtual void reset_sweeps(bool skip_therm) override {
        Base::reset_sweeps(skip_therm);
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

    virtual phase_point phase_space_point() const override {
        return hamiltonian_.phase_space_point();
    }

    virtual bool update_phase_point(phase_point const& pp) override {
        Base::update_phase_point(pp);
        auto const& current_pp = hamiltonian_.phase_space_point();
        bool changed = (current_pp != pp);
        if (changed)
            hamiltonian_.phase_space_point(pp);
        return changed;
    }
};

}
}
