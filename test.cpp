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
#include "embarrassing_adapter.hpp"
#include "filesystem.hpp"
#include "mpi.hpp"
#include "pt_adapter.hpp"
#include "svm-wrapper.hpp"
#include "test_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <tuple>
#include <utility>

#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>
#include <alps/hdf5/multi_array.hpp>

#include <boost/multi_array.hpp>

using sim_type = test_adapter<sim_base>;
using results_type = alps::results_type<sim_type>::type;

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
        auto all_phase_points = [&] {
            std::vector<phase_point> points;
            auto sweep_pol = phase_space::sweep::from_parameters<phase_point>(
                parameters, "test.");
            std::mt19937 rng{parameters["SEED"].as<size_t>() + 1};
            std::generate_n(std::back_inserter(points), sweep_pol->size(),
                [&, p=phase_point{}]() mutable {
                    sweep_pol->yield(p, rng);
                    return p;
                });
            return points;
        }();

        mpi::mutex archive_mutex(comm_world);

        sim_type sim = [&] {
            std::lock_guard<mpi::mutex> archive_guard(archive_mutex);
            return sim_type(parameters, comm_world.rank());
        }();

        // get the bias parameters
        struct skeleton_classifier {
            sim_base::phase_label label1, label2;
            double rho;
        };
        auto classifiers = [&] {
            using kernel_t = svm::kernel::polynomial<2>;
            using phase_label = sim_base::phase_label;
            using model_t = svm::model<kernel_t, phase_label>;

            std::vector<skeleton_classifier> cl;

            if (sim.has_model()) {
                std::string arname = parameters["outputfile"];
                alps::hdf5::archive ar(arname, "r");

                model_t model;
                svm::model_serializer<svm::hdf5_tag, model_t> serial(model);
                ar["model"] >> serial;
                for (auto const& c : model.classifiers())
                    cl.push_back({c.labels().first, c.labels().second, c.rho()});
            }
            return cl;
        }();

        const std::string test_filename = parameters["test.filename"];
        const bool resumed = alps::origin_name(parameters) == test_filename;
        alps::stop_callback stop_cb(parameters["timelimit"].as<size_t>());

        using batches_type = typename sim_type::batcher::batches_type;
        using proxy_t = dispatcher<batches_type>::archive_proxy_type;

        dispatcher<batches_type> dispatch(test_filename,
            archive_mutex,
            resumed,
            sim_type::batcher{parameters}(all_phase_points),
            stop_cb,
            [&](proxy_t ar) { log() << "restoring checkpoint\n"; ar >> sim; },
            [&](proxy_t ar) { log() << "writing checkpoint\n"; ar << sim; });

        std::vector<size_t> available_results;
        if (resumed && is_master) {
            std::lock_guard<mpi::mutex> archive_guard(archive_mutex);
            alps::hdf5::archive ar(test_filename, "r");
            ar["results/available"] >> available_results;
        }

        while (dispatch.request_batch()) {
            bool valid = dispatch.valid();
            auto comm_valid = mpi::split_communicator(dispatch.comm_group, valid);
            if (!valid)
                continue;
            sim.rebind_communicator(comm_valid);
            if (dispatch.point_resumed()) {
                auto slice_point = sim.phase_space_point();
                log() << "resuming batch " << dispatch.batch_index() << ": "
                      << slice_point << std::endl;
            } else {
                auto slice_point = dispatch.point();
                log() << "working on batch " << dispatch.batch_index() << ": "
                      << slice_point << std::endl;
                sim.reset_sweeps(!sim.update_phase_point(slice_point));
            }

            bool finished = sim.run(stop_cb);

            // only process results if batch was completed
            if (finished) {
                int n_points = sim.number_of_points();
                log() << "collecting results..." << std::endl;
                results_type results = alps::collect_results(sim);

                // save the results
                if (comm_valid.rank() < n_points) {
                    // need to refresh slice_point since it may have changed
                    // in the course of the simulation (e.g. PT)
                    auto slice_point = sim.phase_space_point();

                    std::lock_guard<mpi::mutex> archive_guard(archive_mutex);
                    alps::hdf5::archive ar(test_filename, "w");
                    std::stringstream ss;
                    ss << "results/" << dispatch.batch_index() << "/";
                    if (dispatch.is_group_leader)
                        ar[ss.str() + "n_points"] << n_points;
                    ss << comm_valid.rank();
                    ar[ss.str() + "/measurements"] << results;
                    ar[ss.str() + "/point"]
                        << std::vector<double>{slice_point.begin(),
                            slice_point.end()};
                    available_results.push_back(dispatch.batch_index());
                }
            }
        }

        auto gathered_results = [&] {
            std::vector<size_t> res(dispatch.batches.size() * comm_world.size());
            auto end = mpi::all_gather(comm_world,
                available_results.begin(),
                available_results.end(),
                res.begin());
            std::sort(res.begin(), end);
            end = std::unique(res.begin(), end);
            res.resize(end - res.begin());
            return res;
        }();

        if (!is_master)
            return 0;

        std::cout << "results from " << gathered_results.size()
                  << " batches available\n";

        std::lock_guard<mpi::mutex> archive_guard(archive_mutex);
        alps::hdf5::archive ar(test_filename, "w");
        ar["results/available"] << gathered_results;

        std::map<phase_point, results_type> all_results;
        for (size_t batch_index : gathered_results) {
            size_t n_points = [&] {
                std::stringstream ss;
                ss << "results/" << batch_index << "/n_points";
                size_t np;
                ar[ss.str()] >> np;
                return np;
            }();
            for (size_t i = 0; i < n_points; ++i) {
                std::stringstream ss;
                ss << "results/" << batch_index << '/' << i;
                std::vector<double> point_vec;
                ar[ss.str() + "/point"] >> point_vec;
                decltype(all_results)::iterator it;
                bool inserted;
                std::tie(it, inserted) =
                    all_results.emplace(std::piecewise_construct,
                        std::forward_as_tuple(point_vec.begin()),
                        std::forward_as_tuple());
                if (!inserted)
                    std::cout << "warning: duplicate point " << it->first << '\n';
                ar[ss.str() + "/measurements"] >> it->second;
            }
        }

        // rescale the SVM decision function to unit interval
        std::function<void(std::vector<double>&, int, double)> rescale;
        if (cmdl[{"-r", "--rescale"}]) {
            std::vector<double> scale_factors(
                all_results.begin()->second["SVM"].mean<std::vector<double>>().size(),
                1.);
            for (size_t j = 0; j < scale_factors.size(); ++j) {
                double min = std::numeric_limits<double>::max();
                double max = std::numeric_limits<double>::min();
                for (auto const& p : all_results) {
                    auto mean = p.second["SVM"].mean<std::vector<double>>();
                    min = std::min(mean[j], min);
                    max = std::max(mean[j], max);
                }
                scale_factors[j] = 1. / (max - min);
            }
            rescale = [=](std::vector<double>& vec, int exponent, double shift) {
                for (size_t j = 0; j < vec.size(); ++j)
                    vec[j] = pow(scale_factors[j], exponent)
                        * (vec[j] + shift * classifiers[j].rho);
                return vec;
            };
        } else {
            rescale = [](std::vector<double>& vec, int, double) {
                return vec;
            };
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
            for (std::string const& opname : sim.order_param_names())
                annotate(opname);
            if (sim.has_model()) {
                sim_type::phase_classifier phase_classifier(parameters);
                annotate("SVM classification label");
                for (auto const& cl : classifiers) {
                    std::stringstream ss;
                    ss << "decision function "
                       << phase_classifier.name(cl.label1) << " / "
                       << phase_classifier.name(cl.label2);
                    annotate(ss.str());
                    annotate(ss.str() + " variance");
                }
            }

            // data
            for (auto const& p : all_results) {
                phase_point const& pp = p.first;
                results_type const& res = p.second;
                std::copy(pp.begin(), pp.end(),
                    std::ostream_iterator<double>{os, "\t"});
                for (std::string const& opname : sim.order_param_names())
                    os << res[opname].mean<double>() << '\t'
                       << res[opname].error<double>() << '\t';
                if (sim.has_model()) {
                    auto svm_mean = res["SVM"].mean<std::vector<double>>();
                    rescale(svm_mean, 1, 1.);
                    auto svm_error = res["SVM"].error<std::vector<double>>();
                    rescale(svm_error, 1, 0.);
                    auto svm_var = res["SVM^2"] - res["SVM"] * res["SVM"];
                    auto svm_var_mean = svm_var.mean<std::vector<double>>();
                    rescale(svm_var_mean, 2, 0.);
                    auto svm_var_error = svm_var.error<std::vector<double>>();
                    rescale(svm_var_error, 2, 0.);
                    os << res["label"].mean<double>() << '\t'
                       << res["label"].error<double>() << '\t';
                    for (size_t i = 0; i < svm_mean.size(); ++i)
                        os << svm_mean[i] << '\t' << svm_error[i] << '\t'
                           << svm_var_mean[i] << '\t' << svm_var_error[i] << '\t';
                }
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
