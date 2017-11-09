#pragma once

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
            ;
    }

    training_adapter (parameters_type & parms, std::size_t seed_offset = 0)
        : Simulation(parms, seed_offset)
        , temp_step(double(parameters["temp_step"]))
        , temp_crit(double(parameters["temp_crit"]))
        , temp_sigma_sq(pow(double(parameters["temp_sigma"]), 2))
        , temp_min(double(parameters["temp_min"]))
        , temp_max(double(parameters["temp_max"]))
        , N_temp(size_t(parameters["N_temp"]))
        , temp(temp_crit)
        , n_temp(0)
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
            ++n_temp;
        }
        Simulation::update();
    }

    using alps::mcbase::save;
    virtual void save (alps::hdf5::archive & ar) const override {
        Simulation::save(ar);

        // non-overridable parameters
        ar["training/temp_crit"] << temp_crit;

        // state
        ar["training/temp"] << temp;
        ar["training/n_temp"] << n_temp;
    }

    using alps::mcbase::load;
    virtual void load (alps::hdf5::archive & ar) override {
        Simulation::load(ar);

        // non-overridable parameters
        ar["training/temp_crit"] >> temp_crit;

        // state
        ar["training/temp"] >> temp;
        ar["training/n_temp"] >> n_temp;
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

    size_t n_temp;
    double temp;
};
