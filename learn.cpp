/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "checkpointing_stop_callback.hpp"
#include "svm-wrapper.hpp"
#include "training_adapter.hpp"
#include "test_adapter.hpp"

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
using model_t = svm::model<kernel_t>;

int main(int argc, char** argv)
{
    try {
    
        // Creates the parameters for the simulation
        // If an hdf5 file is supplied, reads the parameters there
        std::cout << "Initializing parameters..." << std::endl;

        alps::params parameters(argc, argv);
        sim_type::define_parameters(parameters);
        define_test_parameters(parameters);

        if (!parameters.is_restored()) {
            parameters.define<double>("nu", 0.5, "nu_SVC regularization parameter");
            parameters.define<size_t>("progress_interval", 3,
                                      "time in sec between progress reports");
        }

        if (parameters.help_requested(std::cout) ||
            parameters.has_missing(std::cout)) {
            return 1;
        }

        svm::problem<kernel_t> prob(0);
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
                std::unique_ptr<sim_type> sim;
#pragma omp critical
                sim = std::unique_ptr<sim_type>(new sim_type(parameters, global_progress, tid));

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
                    cp[checkpoint_path] >> *sim;
                }

                auto progress_report = [&] () {
                    double local_frac = sim->local_fraction_completed();
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
                sim->run(checkpointing_stop_callback(size_t(parameters["timelimit"]),
                                                     size_t(parameters["progress_interval"]),
                                                     progress_report));

                // Checkpoint the simulation
#pragma omp critical
                {
                    std::cout << "Checkpointing simulation to " << checkpoint_file
                            << " (thread " << tid << ")"
                            << std::endl;
                    alps::hdf5::archive cp(checkpoint_file, "w");
                    cp[checkpoint_path] << *sim;
                }

#pragma omp critical
                {
                    if (prob.dim() == 0) {
                        prob = sim->surrender_problem();
                    } else {
                        prob.append_problem(sim->surrender_problem());
                    }
                }
            }
        }

        {
            alps::hdf5::archive cp(checkpoint_file, "w");
            cp["simulation/n_clones"] << n_clones;
        }

        {
            // create the model
            svm::parameters<kernel_t> kernel_params(1., 0.,
                                                    parameters["nu"].as<double>(),
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
