/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
            parameters.define<double>("C", 1., "C_SVC regularization parameter");
        }

        if (parameters.help_requested(std::cout) ||
            parameters.has_missing(std::cout)) {
            return 1;
        }

        svm::problem<kernel_t> prob(0);
        std::string checkpoint_file = parameters["checkpoint"].as<std::string>();

        // determine number of temps remaining
        size_t remaining_temps = parameters["N_temp"].as<size_t>();
        size_t temps_done = 0;
        int n_clones;
        if (parameters.is_restored()) {
            alps::hdf5::archive cp(checkpoint_file, "r");
            cp["simulation/temps_done"] >> temps_done;
            remaining_temps -= temps_done;

            cp["simulation/n_clones"] >> n_clones;
        }
    
#pragma omp parallel
        {
            if (!parameters.is_restored()) {
                n_clones = omp_get_num_threads();
            }

#pragma omp for schedule(dynamic)
            for (int tid = 0; tid < n_clones; ++tid) {
                size_t N_temp_thread = remaining_temps / n_clones;
                if (tid == n_clones - 1)
                    N_temp_thread += remaining_temps % n_clones;

                std::unique_ptr<sim_type> sim;
#pragma omp critical
                sim = std::unique_ptr<sim_type>(new sim_type(parameters, N_temp_thread, tid));

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

                sim->run(alps::stop_callback(size_t(parameters["timelimit"])));

                // Checkpoint the simulation
#pragma omp critical
                {
                    std::cout << "Checkpointing simulation to " << checkpoint_file
                            << " (thread " << tid << ")"
                            << std::endl;
                    alps::hdf5::archive cp(checkpoint_file, "w");
                    cp[checkpoint_path] << *sim;
                    temps_done += sim->temps_done();
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
            cp["simulation/temps_done"] << temps_done;
            cp["simulation/n_clones"] << n_clones;
        }

        {
            // create the model
            svm::parameters<kernel_t> kernel_params(1., 0.,
                                                    parameters["C"].as<double>(),
                                                    svm::machine_type::C_SVC);
            std::cout << "Creating SVM model..."
                      << " (C = " << parameters["C"].as<double>() << ')'
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
