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

#include "svm-wrapper.hpp"
#include "test_adapter.hpp"
#include "filesystem.hpp"
#include "argh.h"
#include "override_parameters.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <utility>

#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>
#include <alps/hdf5/multi_array.hpp>

#include <boost/multi_array.hpp>

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

        argh::parser cmdl(argc, argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);
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
            for (auto const& p : parameters) {
                if (p.second.isType<int>())
                    override_parameter<int>(p.first, parameters, cmdl);
                if (p.second.isType<long>())
                    override_parameter<long>(p.first, parameters, cmdl);
                if (p.second.isType<size_t>())
                    override_parameter<size_t>(p.first, parameters, cmdl);
                if (p.second.isType<float>())
                    override_parameter<float>(p.first, parameters, cmdl);
                if (p.second.isType<double>())
                    override_parameter<double>(p.first, parameters, cmdl);
                if (p.second.isType<std::string>())
                    override_parameter<std::string>(p.first, parameters, cmdl);
            }
        }

        using phase_point = sim_base::phase_point;
        auto points = [&parameters] {
            std::vector<phase_point> ps;
            phase_point p;
            ps.reserve(parameters["test.N_scan"].as<size_t>());
            using scan_t = phase_space::sweep::line_scan<phase_point>;
            scan_t scan(phase_point(parameters, "test.a."),
                        phase_point(parameters, "test.b."),
                        parameters["test.N_scan"].as<size_t>());
            while (scan.yield(p)) {
                ps.push_back(p);
            }
            return ps;
        } ();
        using pair_t = std::pair<double, double>;
        using vecpair_t = std::pair<std::vector<double>, std::vector<double>>;
        std::vector<pair_t> label(points.size());
        std::vector<vecpair_t> svm(points.size());
        std::vector<vecpair_t> mag(points.size());

        alps::hdf5::archive ar(parameters["test.filename"].as<std::string>(), "w");
        ar["parameters"] << parameters;

        boost::multi_array<double, 2> phase_points(boost::extents[points.size()][phase_point::label_dim]);
        for (size_t i = 0; i < points.size(); ++i)
            std::copy(points[i].begin(), points[i].end(), phase_points[i].begin());
        ar["phase_points"] << phase_points;

        size_t done = 0;
        std::vector<phase_point> current_points;
#pragma omp parallel
        {
#pragma omp single
            current_points.resize(omp_get_num_threads());
            size_t tid = omp_get_thread_num();
            auto progress_report = [&] {
                size_t now_done;
                now_done = done;
                std::cout << '[' << now_done << '/' << points.size() << "] ";
                for (size_t j = 0; j < current_points.size(); ++j) {
                    phase_point now_point;
                    now_point = current_points[j];
                    std::cout << std::setw(8) << std::setprecision(4)
                              << now_point << ',';
                }
                std::cout << "           \r" << std::flush;
            };
#pragma omp for schedule(dynamic)
            for (size_t i = 0; i < points.size(); ++i) {
                current_points[tid] = points[i];
#pragma omp critical
                progress_report();

                sim_type sim(parameters);
                phase_space::sweep::cycle<phase_point> one_point {{points[i]}};
                sim.update_phase_point(one_point);
                sim.run(alps::stop_callback(size_t(parameters["timelimit"])));

                alps::results_type<sim_type>::type results = alps::collect_results(sim);
                std::stringstream ss;
#pragma omp critical
                {
                    ss << "results/" << i;
                    ar[ss.str()] << results;
                }

                for (std::string const& opname : sim.order_param_names()) {
                    mag[i].first.push_back(results[opname].mean<double>());
                    mag[i].second.push_back(results[opname].error<double>());
                }
                svm[i] = {results["SVM"].mean<std::vector<double>>(),
                          results["SVM"].error<std::vector<double>>()};
                label[i] = {results["label"].mean<double>(),
                            results["label"].error<double>()};
#pragma omp atomic
                ++done;
            }
#pragma omp critical
            progress_report();
        }
        std::cout << std::endl;

        // rescale the SVM decision function to unit interval
        for (size_t j = 0; j < svm.front().first.size(); ++j) {
            double min = std::numeric_limits<double>::max();
            double max = std::numeric_limits<double>::min();
            for (size_t i = 0; i < points.size(); ++i) {
                double x = svm[i].first[j];
                if (x < min)
                    min = x;
                if (x > max)
                    max = x;
            }
            double fac = 1. / (max - min);
            for (size_t i = 0; i < points.size(); ++i) {
                // svm[i].first[j] = fac * (svm[i].first[j] - min);
                svm[i].first[j] *= fac;
                svm[i].second[j] *= fac;
            }
        }

        // output
        std::ofstream os(parameters["test.txtname"].as<std::string>());
        for (size_t i = 0; i < points.size(); ++i) {
            std::copy(points[i].begin(), points[i].end(),
                      std::ostream_iterator<double> {os, "\t"});
            os << label[i].first << '\t'
               << label[i].second << '\t';
            for (size_t j = 0; j < svm[i].first.size(); ++j)
                os << svm[i].first[j] << '\t'
                   << svm[i].second[j] << '\t';
                // os << sqrt(svm[i].first[j]) << '\t'
                //    << svm[i].second[j] / 2. / sqrt(svm[i].first[j]) << '\t';
            for (size_t j = 0; j < mag[i].first.size(); ++j)
                os << mag[i].first[j] << '\t'
                   << mag[i].second[j] << '\t';
            os << '\n';
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
