/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "svm-wrapper.hpp"
#include "training_adapter.hpp"
#include "test_adapter.hpp"

#include <iostream>

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
    typedef training_adapter<sim_base> sim_type;

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
    
        std::cout << "Creating simulation" << std::endl;
        sim_type sim(parameters); 

        // If needed, restore the last checkpoint
        std::string checkpoint_file = parameters["checkpoint"].as<std::string>();
        
        if (parameters.is_restored()) {
            std::cout << "Restoring checkpoint from " << checkpoint_file
                      << std::endl;
            sim.load(checkpoint_file);
        }

        // Run the simulation
        std::cout << "Running simulation" << std::endl;
        sim.run(alps::stop_callback(size_t(parameters["timelimit"])));

        // Checkpoint the simulation
        std::cout << "Checkpointing simulation to " << checkpoint_file
                  << std::endl;
        sim.save(checkpoint_file);

        alps::results_type<sim_type>::type results = alps::collect_results(sim);

        // Print results
        {
            using alps::accumulators::result_wrapper;
            std::cout << "All measured results:" << std::endl;
            std::cout << results << std::endl;
            
            std::cout << "Simulation ran for "
                      << results["Energy"].count()
                      << " steps." << std::endl;

            // create the model
            using kernel_t = typename sim_type::kernel_t;
            using model_t = svm::model<kernel_t>;
            svm::parameters<kernel_t> kernel_params(1., 0.,
                                                    parameters["C"].as<double>(),
                                                    svm::machine_type::C_SVC);
            std::cout << "Creating SVM model..."
                      << " (C = " << parameters["C"].as<double>() << ')'
                      << std::endl;
            model_t model(sim.surrender_problem(), kernel_params);

            // set up serializer
            svm::model_serializer<svm::hdf5_tag, model_t> serial(model);
            
            // Saving to the output file
            std::string output_file = parameters["outputfile"];
            alps::hdf5::archive ar(output_file, "w");
            ar["/parameters"] << parameters;
            ar["/simulation/results"] << results;
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
