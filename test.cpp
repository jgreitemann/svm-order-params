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

#include "argh.h"
#include "config_sim_base.hpp"
#include "dispatcher.hpp"
#include "filesystem.hpp"
#include "mpi.hpp"
#include "svm-wrapper.hpp"
#include "test_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>

#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>
#include <alps/hdf5/multi_array.hpp>

#include <boost/multi_array.hpp>

using sim_type = test_adapter<sim_base>;

int main(int argc, char** argv)
{
    mpi::environment env(argc, argv, mpi::environment::threading::multiple);
    mpi::communicator comm_world;

    const bool is_master = (comm_world.rank() == 0);
    try {
        // Creates the parameters for the simulation
        // If an hdf5 file is supplied, reads the parameters there
        if (is_master)
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

        if (parameters.help_requested(std::cout)
            || parameters.has_missing(std::cout))
        {
            return 1;
        }

        auto log = [&](std::ostream & os = std::cout) -> std::ostream & {
            return os << '[' << comm_world.rank() << '/' << comm_world.size()
                      << "]\t";
        };

        // Collect phase points
        using phase_point = sim_base::phase_point;
        using batches_type = std::vector<std::vector<phase_point>>;
        auto get_batches = [&] {
            using scan_t = typename sim_base::test_sweep_type;
            batches_type batches;
            scan_t scan{parameters, 0, "test."};
            std::generate_n(std::back_inserter(batches), scan.size(),
                [&, p=phase_point{}]() mutable -> std::vector<phase_point> {
                    scan.yield(p);
                    return {p};
                });
            return batches;
        };

        using pair_t = std::pair<double, double>;
        using vecpair_t = std::pair<std::vector<double>, std::vector<double>>;
        std::vector<phase_point> points;
        std::vector<pair_t> label;
        std::vector<vecpair_t> svm;
        std::vector<vecpair_t> svm_var;
        std::vector<vecpair_t> mag;

        // get the bias parameters
        struct skeleton_classifier {
            sim_base::phase_label label1, label2;
            double rho;
        };
        auto classifiers = [&] {
            using kernel_t = svm::kernel::polynomial<2>;
            using phase_label = sim_base::phase_label;
            using model_t = svm::model<kernel_t, phase_label>;

            std::string arname = parameters["outputfile"];
            alps::hdf5::archive ar(arname, "r");

            model_t model;
            svm::model_serializer<svm::hdf5_tag, model_t> serial(model);
            ar["model"] >> serial;
            std::vector<skeleton_classifier> cl;
            for (auto const& c : model.classifiers())
                cl.push_back({c.labels().first, c.labels().second, c.rho()});
            return cl;
        }();

        mpi::mutex archive_mutex(comm_world);

        sim_type sim = [&] {
            std::lock_guard<mpi::mutex> archive_guard(archive_mutex);
            return sim_type(parameters, comm_world.rank());
        }();

        const std::string test_filename = parameters["test.filename"];
        const bool resumed = alps::origin_name(parameters) == test_filename;
        alps::stop_callback stop_cb(parameters["timelimit"].as<size_t>());

        using proxy_t = dispatcher<batches_type>::archive_proxy_type;

        dispatcher<batches_type> dispatch(test_filename,
            archive_mutex,
            resumed,
            get_batches(),
            stop_cb,
            [&](proxy_t ar) { log() << "restoring checkpoint\n"; ar >> sim; },
            [&](proxy_t ar) { log() << "writing checkpoint\n"; ar << sim; });

        while (dispatch.request_batch()) {
            if (!dispatch.valid())
                continue;
            auto slice_point = dispatch.point();
            log() << "working on batch " << dispatch.batch_index() << ": "
                  << slice_point << '\n';
            if (dispatch.point_resumed()) {
                if (slice_point != sim.phase_space_point()) {
                    std::stringstream ss;
                    ss << "Inconsistent phase space point found when restoring "
                       << " from checkpoint: expected " << sim.phase_space_point()
                       << ", found " << slice_point << ".";
                    throw std::runtime_error(ss.str());
                }
            } else {
                sim.update_phase_point(slice_point);
            }
            bool finished = sim.run(stop_cb);

            // only process results if batch was completed
            if (finished) {
                alps::results_type<sim_type>::type results = alps::collect_results(sim);

                // save the results
                {
                    std::lock_guard<mpi::mutex> archive_guard(archive_mutex);
                    alps::hdf5::archive ar(test_filename, "w");
                    std::stringstream ss;
                    ss << "results/" << slice_point;
                    log() << "writing " << ss.str() << '\n';
                    ar[ss.str()] << results;
                }

                // log statistics
                points.push_back(slice_point);
                mag.emplace_back();
                for (std::string const& opname : sim.order_param_names()) {
                    mag.back().first.push_back(results[opname].mean<double>());
                    mag.back().second.push_back(results[opname].error<double>());
                }
                auto variance = results["SVM^2"] - results["SVM"] * results["SVM"];
                svm.emplace_back(results["SVM"].mean<std::vector<double>>(),
                    results["SVM"].error<std::vector<double>>());
                svm_var.emplace_back(variance.mean<std::vector<double>>(),
                    variance.error<std::vector<double>>());
                label.emplace_back(results["label"].mean<double>(),
                    results["label"].error<double>());
            }
        }

        // gather results here!

        if (!is_master)
            return 0;

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
            for (std::string const& opname : sim.order_param_names())
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
