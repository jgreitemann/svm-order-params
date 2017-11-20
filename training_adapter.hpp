#pragma once

#include "svm-wrapper.hpp"
#include "hdf5_serialization.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <alps/mc/mcbase.hpp>


template <class Simulation>
class training_adapter : public Simulation {
public:
    typedef alps::mcbase::parameters_type parameters_type;

    using kernel_t = svm::kernel::polynomial<2>;
    using problem_t = svm::problem<kernel_t>;

    static void define_parameters(parameters_type & parameters) {
        // If the parameters are restored, they are already defined
        if (parameters.is_restored()) {
            return;
        }
        
        // Adds the parameters of the base class
        Simulation::define_parameters(parameters);
        parameters
            .define<double>("temp_step", 0.25, "maximum change of temperature")
            .define<double>("temp_crit", 2.269185, "critical temperature")
            .define<double>("temp_sigma", 1.0, "std. deviation of temperature")
            .define<double>("temp_min", 0.0, "minimum value of temperature")
            .define<double>("temp_max", std::numeric_limits<double>::max(),
                            "maximum value of temperature")
            .define<size_t>("N_temp", 1000, "number of attempted temperature updates")
            .define<size_t>("N_sample", 1000, "number of configuration samples taken"
                            " at each temperature")
            ;
    }

    training_adapter (parameters_type & parms,
                      size_t N_temp,
                      std::size_t seed_offset = 0)
        : Simulation(parms, seed_offset)
        , temp_step(double(parameters["temp_step"]))
        , temp_crit(double(parameters["temp_crit"]))
        , temp_sigma_sq(pow(double(parameters["temp_sigma"]), 2))
        , temp_min(double(parameters["temp_min"]))
        , temp_max(double(parameters["temp_max"]))
        , N_temp(N_temp)
        , N_sample(size_t(parameters["N_sample"]))
        , temp(temp_crit)
        , n_temp(0)
        , i_temp(0)
        , problem(Simulation::configuration_size())
        , prob_serializer(problem)
    {
        update_temperature();
    }

    virtual double fraction_completed() const {
        return (n_temp + Simulation::fraction_completed()) / N_temp;
    }

    virtual void update () override {
        if (Simulation::fraction_completed() >= 1.) {
            bool changed = update_temperature();
            Simulation::reset_sweeps(!changed);
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
            std::cout << "take sample at temp T = " << temp
                      << " (" << n_temp << ", frac = " << frac << ')' << std::endl;
            problem.add_sample(Simulation::configuration(), order_label());
            ++i_temp;
        }
    }

    double order_label () const {
        return (temp < temp_crit) ? 1. : -1.;
    }

    using alps::mcbase::save;
    virtual void save (alps::hdf5::archive & ar) const override {
        Simulation::save(ar);

        // non-overridable parameters
        ar["training/temp_crit"] << temp_crit;
        ar["training/N_sample"] << N_sample;

        // state
        ar["training/temp"] << temp;
        ar["training/i_temp"] << i_temp;

        ar["training/problem"] << prob_serializer;
    }

    using alps::mcbase::load;
    virtual void load (alps::hdf5::archive & ar) override {
        Simulation::load(ar);

        // non-overridable parameters
        ar["training/temp_crit"] >> temp_crit;
        ar["training/N_sample"] >> N_sample;

        // state
        ar["training/temp"] >> temp;
        ar["training/i_temp"] >> i_temp;

        ar["training/problem"] >> prob_serializer;
        if (problem.dim() != Simulation::configuration_size())
            throw std::runtime_error("invalid problem dimension");

        // flatten
        if (i_temp == N_sample) {
            bool changed = update_temperature();
            Simulation::reset_sweeps(!changed);
            i_temp = 0;
        }
    }

    problem_t surrender_problem () {
        problem_t other_problem (Simulation::configuration_size());
        std::swap(other_problem, problem);
        return other_problem;
    }

    size_t temps_done () const {
        return n_temp + (i_temp == N_sample);
    }

private:
    bool update_temperature () {
        double delta_temp = (2. * random() - 1.) * temp_step;
        if (temp + delta_temp < temp_min || temp + delta_temp > temp_max)
            return false;
        double ratio = exp(-(2. * (temp-temp_crit) + delta_temp) * delta_temp
                           / 2. / temp_sigma_sq);
        if (ratio > 1 || random() < ratio) {
            temp += delta_temp;
            Simulation::temperature(temp);
            return true;
        }
        return false;
    }

    using Simulation::parameters;
    using Simulation::random;

    double temp_step;
    double temp_crit;
    double temp_sigma_sq;
    double temp_min;
    double temp_max;
    size_t N_temp;
    size_t N_sample;

    size_t n_temp;
    size_t i_temp;
    double temp;

    problem_t problem;
    svm::problem_serializer<svm::hdf5_tag, problem_t> prob_serializer;
};
