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

#include "config_sim_base.hpp"
#include "svm-wrapper.hpp"
#include "test_adapter.hpp"
#include "filesystem.hpp"
#include "argh.h"

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


int main(int argc, char** argv)
{
    typedef test_adapter<sim_base> sim_type;

    try {

        // Creates the parameters for the simulation
        // If an hdf5 file is supplied, reads the parameters there
        std::cout << "Initializing parameters..." << std::endl;
        alps::params parameters(argc, argv);
        argh::parser cmdl(argc, argv);

        // define parameters
        sim_type::define_parameters(parameters);
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

        using phase_point = sim_base::phase_point;
        auto points = [&parameters] {
            using scan_t = typename sim_base::test_sweep_type;
            scan_t scan{parameters, 0, "test."};
            std::vector<phase_point> ps(scan.size());
            std::generate(ps.begin(), ps.end(),
                          [p=phase_point{}, &scan]() mutable {
                              scan.yield(p);
                              return p;
                          });
            return ps;
        } ();
        using pair_t = std::pair<double, double>;
        using vecpair_t = std::pair<std::vector<double>, std::vector<double>>;
        std::vector<pair_t> label(points.size());
        std::vector<vecpair_t> svm(points.size());
        std::vector<vecpair_t> svm_var(points.size());
        std::vector<vecpair_t> mag(points.size());

        // get the bias parameters
        struct skeleton_classifier {
            sim_base::phase_label label1, label2;
            double rho;
        };
        auto classifiers = [&] {
            using kernel_t = svm::kernel::polynomial<2>;
            using phase_label = sim_base::phase_label;
            using model_t = svm::model<kernel_t, phase_label>;

            std::string arname = parameters.get_archive_name();
            alps::hdf5::archive ar(arname, "r");

            model_t model;
            svm::model_serializer<svm::hdf5_tag, model_t> serial(model);
            ar["model"] >> serial;
            std::vector<skeleton_classifier> cl;
            for (auto const& c : model.classifiers())
                cl.push_back({c.labels().first, c.labels().second, c.rho()});
            return cl;
        }();

        alps::hdf5::archive ar(parameters["test.filename"].as<std::string>(), "w");
        ar["parameters"] << parameters;

        boost::multi_array<double, 2> phase_points(boost::extents[points.size()][phase_point::label_dim]);
        for (size_t i = 0; i < points.size(); ++i)
            std::copy(points[i].begin(), points[i].end(), phase_points[i].begin());
        ar["phase_points"] << phase_points;

        size_t done = 0;
        std::vector<phase_point> current_points;
        std::vector<std::string> order_param_names;
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

                if (i == 0)
                    order_param_names = sim.order_param_names();
                for (std::string const& opname : sim.order_param_names()) {
                    mag[i].first.push_back(results[opname].mean<double>());
                    mag[i].second.push_back(results[opname].error<double>());
                }
                auto variance = results["SVM^2"] - results["SVM"] * results["SVM"];
                svm[i] = {results["SVM"].mean<std::vector<double>>(),
                          results["SVM"].error<std::vector<double>>()};
                svm_var[i] = {variance.mean<std::vector<double>>(),
                              variance.error<std::vector<double>>()};
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
        if (cmdl[{"-r", "--rescale"}]) {
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
                    svm[i].first[j] += classifiers[j].rho;
                    svm[i].first[j] *= fac;
                    svm[i].second[j] *= fac;
                }
            }
        }

        // output
        {
            std::ofstream os(parameters["test.txtname"].as<std::string>());

            // column key
            size_t k = phase_point::label_dim;
            os << "# column(s): quantity\n";
            os << "# 1-" << k++ << ": phase space point\n";
            auto annotate = [&] (std::string const& name) {
                os << "# " << k << " (" << (k + 1) << "): "
                   << name << " (error)\n";
                k += 2;
            };
            annotate("SVM classification label");
            for (auto const& cl : classifiers) {
                std::stringstream ss;
                ss << "decision function " << cl.label1 << " / " << cl.label2;
                annotate(ss.str());
                annotate(ss.str() + " variance");
            }
            for (std::string const& opname : order_param_names)
                annotate(opname);

            // data
            for (size_t i = 0; i < points.size(); ++i) {
                std::copy(points[i].begin(), points[i].end(),
                          std::ostream_iterator<double> {os, "\t"});
                os << label[i].first << '\t'
                   << label[i].second << '\t';
                for (size_t j = 0; j < svm[i].first.size(); ++j) {
                    os << svm[i].first[j] << '\t'
                       << svm[i].second[j] << '\t';
                    os << svm_var[i].first[j] << '\t'
                       << svm_var[i].second[j] << '\t';
                }
                for (size_t j = 0; j < mag[i].first.size(); ++j)
                    os << mag[i].first[j] << '\t'
                       << mag[i].second[j] << '\t';
                os << '\n';
            }
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
