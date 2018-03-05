#include "svm-wrapper.hpp"
#include "test_adapter.hpp"
#include "filesystem.hpp"
#include "argh.h"
#include "override_parameters.hpp"

#include <omp.h>

#include <cmath>
#include <iomanip>
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

        argh::parser cmdl({ "timelimit", "total_sweeps", "thermalization_sweeps",
                    "sweep_unit", "test.temp_min", "test.temp_max", "test.N_temp",
                    "test.filename", "test.txtname", "SEED" });
        cmdl.parse(argc, argv);
        alps::params parameters = [&] {
            if (cmdl[1].empty())
                return alps::params(argc, argv);
            std::string pseudo_args[] = {cmdl[0], cmdl[1]};
            if (cmdl[{"-h", "--help"}])
                pseudo_args[1] = "--help";
            char const * pseudo_argv[] = {pseudo_args[0].c_str(), pseudo_args[1].c_str()};
            return alps::params(2, pseudo_argv);
        } ();
        sim_type::define_parameters(parameters);
        define_test_parameters(parameters);

        if (parameters["test.filename"].as<std::string>().empty())
            parameters["test.filename"] =
                replace_extension(alps::origin_name(parameters), ".test.h5");
        if (parameters["test.txtname"].as<std::string>().empty())
            parameters["test.txtname"] =
                replace_extension(alps::origin_name(parameters), ".test.txt");

        if (parameters.help_requested(std::cout) ||
            parameters.has_missing(std::cout)) {
            return 1;
        }

        /* WORKAROUND: override parameters from CL args manually */ {
            override_parameter<size_t>("timelimit", parameters, cmdl);
            override_parameter<size_t>("total_sweeps", parameters, cmdl);
            override_parameter<size_t>("thermalization_sweeps", parameters, cmdl);
            override_parameter<size_t>("sweep_unit", parameters, cmdl);
            override_parameter<long>("SEED", parameters, cmdl);

            override_parameter<double>("test.temp_min", parameters, cmdl);
            override_parameter<double>("test.temp_max", parameters, cmdl);
            override_parameter<size_t>("test.N_temp", parameters, cmdl);
            override_parameter<std::string>("test.filename", parameters, cmdl);
            override_parameter<std::string>("test.txtname", parameters, cmdl);
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

        size_t done = 0;
        std::vector<double> current_temps;
#pragma omp parallel
        {
#pragma omp single
            current_temps.resize(omp_get_num_threads(), -1.);
            size_t tid = omp_get_thread_num();
            auto progress_report = [&] {
                size_t now_done;
#pragma omp atomic read
                now_done = done;
                std::cout << '[' << now_done << '/' << temps.size() << "] T = ";
                for (size_t j = 0; j < current_temps.size(); ++j) {
                    double now_temp;
#pragma omp atomic read
                    now_temp = current_temps[j];
                    std::cout << std::setw(8) << std::setprecision(4)
                              << now_temp << ',';
                }
                std::cout << "           \r" << std::flush;
            };
#pragma omp for schedule(dynamic)
            for (size_t i = 0; i < temps.size(); ++i) {
#pragma omp atomic write
                current_temps[tid] = temps[i];
#pragma omp critical
                progress_report();

                alps::params local_params(parameters);
                local_params["temperature"] = temps[i];

                sim_type sim(local_params);
                sim.run(alps::stop_callback(size_t(parameters["timelimit"])));

                alps::results_type<sim_type>::type results = alps::collect_results(sim);
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

#pragma omp atomic
                ++done;
            }
#pragma omp critical
            progress_report();
        }
        std::cout << std::endl;

        // rescale the SVM order parameter to match magnetization at end points
        if (cmp_true) {
            double fac = ((pow(mag.front().first, 2)- pow(mag.back().first, 2))
                          / (svm.front().first - svm.back().first));
            double offset = pow(mag.front().first, 2) - fac * svm.front().first;
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
               << sqrt(svm[i].first) << '\t'
               << svm[i].second / 2. / sqrt(svm[i].first) << '\t'
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
