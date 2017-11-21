/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "svm-wrapper.hpp"
#include "test_adapter.hpp"

#include <omp.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>

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

        if (parameters["test.filename"].as<std::string>().empty())
            parameters["test.filename"] =
                alps::fs::remove_extensions(alps::fs::get_basename(parameters.get_origin_name()))
                + ".test.h5";
        if (parameters["test.txtname"].as<std::string>().empty())
            parameters["test.txtname"] =
                alps::fs::remove_extensions(alps::fs::get_basename(parameters.get_origin_name()))
                + ".test.txt";

        if (parameters.help_requested(std::cout) ||
            parameters.has_missing(std::cout)) {
            return 1;
        }

        std::vector<double> temps(parameters["test.N_temp"].as<size_t>());
        using pair_t = std::pair<double,double>;
        std::vector<pair_t> mag(temps.size());
        std::vector<pair_t> svm(temps.size());
        std::vector<pair_t> ordered(temps.size());
        for (size_t i = 0; i < temps.size(); ++i) {
            double x = 1. * i / (temps.size() - 1);
            temps[i] = (x * parameters["test.temp_max"].as<double>()
                        + (1-x) * parameters["test.temp_min"].as<double>());
        }

        std::string order_param_name = sim_type::order_param_name;
        bool cmp_true = !order_param_name.empty();

        alps::hdf5::archive ar(parameters["test.filename"].as<std::string>(), "w");
        ar["parameters"] << parameters;
        ar["temperatures"] << temps;

#pragma omp parallel for
        for (size_t i = 0; i < temps.size(); ++i) {
            alps::params local_params(parameters);
            local_params["temperature"] = temps[i];

            std::unique_ptr<sim_type> sim;
#pragma omp critical
            sim = std::unique_ptr<sim_type>(new sim_type(local_params));
            sim->run(alps::stop_callback(size_t(parameters["timelimit"])));

            alps::results_type<sim_type>::type results = alps::collect_results(*sim);
            std::stringstream ss;
#pragma omp critical
            {
                ss << "results/" << i;
                ar[ss.str()] << results;
            }

            if (cmp_true) {
                mag[i] = {results[order_param_name].mean<double>(),
                          results[order_param_name].error<double>()};
            }
            svm[i] = {results["SVM"].mean<double>(),
                      results["SVM"].error<double>()};
            ordered[i] = {results["ordered"].mean<double>(),
                          results["ordered"].error<double>()};
        }

        // rescale the SVM order parameter to match magnetization at end points
        if (cmp_true) {
            double fac = ((mag.front().first - mag.back().first)
                          / (svm.front().first - svm.back().first));
            double offset = mag.front().first - fac * svm.front().first;
            for (size_t i = 0; i < temps.size(); ++i) {
                svm[i].first = fac * svm[i].first + offset;
                svm[i].second = std::abs(fac) * svm[i].second;
            }
        }

        // output
        std::ofstream os(parameters["test.txtname"].as<std::string>());
        for (size_t i = 0; i < temps.size(); ++i) {
            os << temps[i] << '\t'
               << ordered[i].first << '\t'
               << ordered[i].second << '\t'
               << svm[i].first << '\t'
               << svm[i].second << '\t'
               << mag[i].first << '\t'
               << mag[i].second << '\n';
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
