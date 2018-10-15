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

#include "svm-wrapper.hpp"
#include "hdf5_serialization.hpp"
#include "phase_space_policy.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <alps/mc/mcbase.hpp>


template <class Simulation>
class training_adapter : public Simulation {
public:
    typedef alps::mcbase::parameters_type parameters_type;

    using phase_point = typename Simulation::phase_point;
    using phase_classifier = typename Simulation::phase_classifier;
    using phase_sweep_policy_type = phase_space::sweep::policy<phase_point>;

    using kernel_t = svm::kernel::polynomial<2>;
    using problem_t = svm::problem<kernel_t, phase_point>;

    static void define_parameters(parameters_type & parameters) {
        // If the parameters are restored, they are already defined
        if (parameters.is_restored()) {
            return;
        }
        
        // Adds the parameters of the base class
        Simulation::define_parameters(parameters);
        phase_classifier::define_parameters(parameters);
        phase_space::sweep::define_parameters<phase_point>(parameters);
        parameters
            .define<std::string>("sweep.dist", "cycle",
                                 "phase space point distribution")
            .define<size_t>("sweep.N", 1000, "number of attempted phase point updates")
            .define<size_t>("sweep.samples", 1000,
                            "number of configuration samples taken"
                            " at each phase point")
            ;
    }

    training_adapter (parameters_type & parms,
                      double const& global_progress,
                      std::size_t seed_offset = 0)
        : Simulation(parms, seed_offset)
        , global_progress(global_progress)
        , N_phase(size_t(parameters["sweep.N"]))
        , N_sample(size_t(parameters["sweep.samples"]))
        , problem(Simulation::configuration_size())
        , prob_serializer(problem)
        , sweep_policy([&] () -> phase_sweep_policy_type * {
                std::string dist_name = parameters["sweep.dist"];
                if (std::is_same<phase_point, phase_space::point::temperature>::value) {
                    if (dist_name == "gaussian")
                        return dynamic_cast<phase_sweep_policy_type*>(
                            new phase_space::sweep::gaussian_temperatures(parms));
                    if (dist_name == "uniform")
                        return dynamic_cast<phase_sweep_policy_type*>(
                            new phase_space::sweep::uniform_temperatures(parms));
                    if (dist_name == "bimodal")
                        return dynamic_cast<phase_sweep_policy_type*>(
                            new phase_space::sweep::equidistant_temperatures(parms, 2, seed_offset));
                }
                if (dist_name == "cycle")
                    return dynamic_cast<phase_sweep_policy_type*>(
                        new phase_space::sweep::cycle<phase_point> (parms, seed_offset));
                if (dist_name == "grid")
                    return dynamic_cast<phase_sweep_policy_type*>(
                        new phase_space::sweep::grid<phase_point> (parms, seed_offset));
                if (dist_name == "uniform")
                    return dynamic_cast<phase_sweep_policy_type*>(
                        new phase_space::sweep::uniform<phase_point> (parms));
                if (dist_name == "uniform_line")
                    return dynamic_cast<phase_sweep_policy_type*>(
                        new phase_space::sweep::uniform_line<phase_point> (parms));
                throw std::runtime_error("Invalid sweep policy \"" + dist_name + "\"");
                return nullptr;
            }())
    {
        Simulation::update_phase_point(*sweep_policy);
    }

    double local_fraction_completed() const {
        return (n_temp + Simulation::fraction_completed()) / N_phase;
    }

    virtual double fraction_completed() const {
        double atomic_progress;
#pragma omp atomic read
        atomic_progress = global_progress;
        return atomic_progress;
    }

    virtual void update () override {
        if (Simulation::fraction_completed() >= 1.) {
            Simulation::update_phase_point(*sweep_policy);
            i_temp = 0;
            ++n_temp;
        }
        Simulation::update();
    }

    virtual void measure () override {
        double frac = Simulation::fraction_completed();
        Simulation::measure();
        if (frac == 0.) return;
        if (frac + 1e-3 >= 1. * (i_temp + 1) / N_sample) {
            problem.add_sample(Simulation::configuration(),
                               Simulation::phase_space_point());
            ++i_temp;
        }
    }

    using alps::mcbase::save;
    virtual void save (alps::hdf5::archive & ar) const override {
        Simulation::save(ar);

        // non-overridable parameters
        ar["training/N_sample"] << N_sample;

        // state
        ar["training/i_temp"] << i_temp;
        ar["training/n_temp"] << n_temp;
        ar["training/sweep"] << *sweep_policy;

        ar["training/problem"] << prob_serializer;
    }

    using alps::mcbase::load;
    virtual void load (alps::hdf5::archive & ar) override {
        Simulation::load(ar);

        // non-overridable parameters
        ar["training/N_sample"] >> N_sample;

        // state
        ar["training/i_temp"] >> i_temp;
        ar["training/n_temp"] >> n_temp;
        ar["training/sweep"] >> *sweep_policy;

        ar["training/problem"] >> prob_serializer;
        if (problem.dim() != Simulation::configuration_size())
            throw std::runtime_error("invalid problem dimension");

        // flatten
        if (i_temp == N_sample) {
            Simulation::update_phase_point(*sweep_policy);
            i_temp = 0;
        }
    }

    problem_t surrender_problem () {
        problem_t other_problem (Simulation::configuration_size());
        std::swap(other_problem, problem);
        return other_problem;
    }

private:

    using Simulation::parameters;
    using Simulation::random;

    double const& global_progress;

    size_t N_phase;
    size_t N_sample;

    size_t n_temp = 0;
    size_t i_temp = 0;

    problem_t problem;
    svm::problem_serializer<svm::hdf5_tag, problem_t> prob_serializer;

    std::unique_ptr<phase_sweep_policy_type> sweep_policy;
};
