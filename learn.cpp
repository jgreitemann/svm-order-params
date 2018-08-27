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

#include "checkpointing_stop_callback.hpp"
#include "svm-wrapper.hpp"
#include "training_adapter.hpp"
#include "test_adapter.hpp"
#include "argh.h"
#include "override_parameters.hpp"

#include <iostream>
#include <memory>
#include <utility>

#include <omp.h>

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


using sim_type = training_adapter<sim_base>;
using kernel_t = typename sim_type::kernel_t;
using label_t = typename sim_type::phase_label;
using classifier_t = typename sim_type::phase_classifier;
using model_t = svm::model<kernel_t, label_t>;
using problem_t = typename model_t::problem_t;

int main(int argc, char** argv)
{
    try {
    
        // Creates the parameters for the simulation
        // If an hdf5 file is supplied, reads the parameters there
        std::cout << "Initializing parameters..." << std::endl;

        argh::parser cmdl({"--nu", "--outputfile", "--timelimit", "--sweep.N",
                           "--sweep.samples"});
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

        if (!parameters.is_restored()) {
            parameters.define<double>("nu", 0.5, "nu_SVC regularization parameter");
            parameters.define<size_t>("progress_interval", 3,
                                      "time in sec between progress reports");
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

        bool skip_sampling = cmdl[{"-s", "--skip-sampling"}];

        if (parameters.help_requested(std::cout) ||
            parameters.has_missing(std::cout)) {
            return 1;
        }

        problem_t prob(0);
        classifier_t classifier(parameters);

        std::string checkpoint_file = parameters["checkpoint"].as<std::string>();

        int n_clones;
        if (parameters.is_restored()) {
            alps::hdf5::archive cp(checkpoint_file, "r");

            cp["simulation/n_clones"] >> n_clones;
        }
    
        std::vector<double> progress;
        double global_progress = 0.;
#pragma omp parallel
        {
            if (!parameters.is_restored()) {
                n_clones = omp_get_num_threads();
            }
#pragma omp master
            progress.resize(n_clones);

#pragma omp for schedule(dynamic)
            for (int tid = 0; tid < n_clones; ++tid) {
                sim_type sim(parameters, global_progress, tid);

                std::string checkpoint_path = [&] {
                    std::stringstream ss;
                    ss << "simulation/clones/" << tid;
                    return ss.str();
                } ();

                // If needed, restore the last checkpoint
#pragma omp critical
                if (parameters.is_restored()) {
                    std::cout << "Restoring checkpoint from " << checkpoint_file
                            << " (thread " << tid << ")"
                            << std::endl;
                    alps::hdf5::archive cp(checkpoint_file, "r");
                    cp[checkpoint_path] >> sim;
                }

                if (!skip_sampling) {
                    auto progress_report = [&] () {
                        double local_frac = sim.local_fraction_completed();
#pragma omp atomic write
                        progress[tid] = local_frac;
#pragma omp master
                        {
                            double total = std::accumulate(progress.begin(),
                                                           progress.end(), 0.);
#pragma omp atomic write
                            global_progress = total;
                            std::cout << std::setprecision(3)
                                      << 100.*total << " %     \r"
                                      << std::flush;
                        }

                    };
                    sim.run(checkpointing_stop_callback(size_t(parameters["timelimit"]),
                                                        size_t(parameters["progress_interval"]),
                                                        progress_report));

                    // Checkpoint the simulation
#pragma omp critical
                    {
                        std::cout << "Checkpointing simulation to " << checkpoint_file
                                  << " (thread " << tid << ")"
                                  << std::endl;
                        alps::hdf5::archive cp(checkpoint_file, "w");
                        cp[checkpoint_path] << sim;
                    }
                }

#pragma omp critical
                {
                    if (prob.dim() == 0) {
                        prob = problem_t(sim.surrender_problem(), classifier);
                    } else {
                        prob.append_problem(sim.surrender_problem(), classifier);
                    }
                }
            }
        }

        /* print label statistics */ {
            std::map<label_t, size_t> label_stat;
            label_t l;
            for (size_t i = 0; i < prob.size(); ++i) {
                std::tie(std::ignore, l) = prob[i];
                auto it = label_stat.find(l);
                if (it != label_stat.end())
                    ++(it->second);
                else
                    label_stat.insert({l, 1});
            }
            std::cout << "\nLabel statistics:\n";
            for (auto const& p : label_stat) {
                std::cout << p.first << ": " << p.second << '\n';
            }
            std::cout << std::endl;
        }

        if (!skip_sampling) {
            alps::hdf5::archive cp(checkpoint_file, "w");
            cp["simulation/n_clones"] << n_clones;
        }

        {
            // create the model
            svm::parameters<kernel_t> kernel_params(parameters["nu"].as<double>(),
                                                    svm::machine_type::NU_SVC);
            std::cout << "Creating SVM model..."
                      << " (nu = " << parameters["nu"].as<double>()
                      << ", total samples = " << prob.size() << ')'
                      << std::endl;
            model_t model(std::move(prob), kernel_params);

            // set up serializer
            svm::model_serializer<svm::hdf5_tag, model_t> serial(model);
            
            // Saving to the output file
            std::string output_file = parameters["outputfile"];
            alps::hdf5::archive ar(output_file, "w");
            ar["/parameters"] << parameters;
            ar["/model"] << serial;
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
