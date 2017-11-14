/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "svm-wrapper.hpp"
#include "test_adapter.hpp"

#include <iostream>
#include <sstream>

#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>

#ifdef ISING
    #include "ising.hpp"
    using sim_base = ising_sim;
#else
#ifdef GAUGE
    #include "gauge.hpp"
    using sim_base = gauge_sim;
#else
    #error Unknown model
#endif
#endif


int main(int argc, char** argv)
{
    typedef test_adapter<sim_base> sim_type;

    try {
    
        // Creates the parameters for the simulation
        // If an hdf5 file is supplied, reads the parameters there
        std::cout << "Initializing parameters..." << std::endl;

        alps::params parameters(argc, argv);
        sim_type::define_parameters(parameters);
        define_test_parameters(parameters);

        if (parameters.help_requested(std::cout) ||
            parameters.has_missing(std::cout)) {
            return 1;
        }

        std::vector<double> temps(parameters["test.N_temp"].as<size_t>());
        for (size_t i = 0; i < temps.size(); ++i) {
            double x = 1. * i / (temps.size() - 1);
            temps[i] = (x * parameters["test.temp_max"].as<double>()
                        + (1-x) * parameters["test.temp_min"].as<double>());
        }

        alps::hdf5::archive ar(parameters["test.filename"].as<std::string>(), "w");
        ar["parameters"] << parameters;
        ar["temperatures"] << temps;

        for (size_t i = 0; i < temps.size(); ++i) {
            alps::params local_params(parameters);
            local_params["temperature"] = temps[i];

            sim_type sim(local_params); 
            sim.run(alps::stop_callback(size_t(parameters["timelimit"])));

            // attention: accumulator_set not threadsafe!
            alps::results_type<sim_type>::type results = alps::collect_results(sim);
            std::stringstream ss;
            ss << "results/" << i;
            ar[ss.str()] << results;

            std::cout << temps[i] << '\t'
                      << results["Magnetization^2"].mean<double>() << ' '
                      << results["Magnetization^2"].error<double>() << '\t'
                      << results["SVM"].mean<double>() << ' '
                      << results["SVM"].error<double>() << std::endl;
        }

        return 0;
    } catch (const std::runtime_error& exc) {
        std::cout << "Exception caught: " << exc.what() << std::endl;
        return 2;
    } catch (...) {
        std::cout << "Unknown exception caught." << std::endl;
        return 2;
    }
}
