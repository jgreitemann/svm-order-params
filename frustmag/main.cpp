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

#include "config_frustmag_sim.hpp"

#include <alps/mc/stop_callback.hpp>

#include <algorithm>
#include <iterator>
#include <iostream>

int main(int argc, char** argv)
{
    // Creates the parameters for the simulation
    // If an hdf5 file is supplied, reads the parameters there
    std::cout << "Initializing parameters..." << std::endl;
    alps::params parameters(argc, argv);

    // define parameters
    sim_base::define_parameters(parameters);

    if (parameters.help_requested(std::cout) ||
        parameters.has_missing(std::cout)) {
        return 1;
    }

    std::string checkpoint_file = parameters["checkpoint"].as<std::string>();

    sim_base sim(parameters);

    std::string checkpoint_path = "simulation/clones/0";

    // If needed, restore the last checkpoint
    if (parameters.is_restored()) {
        std::cout << "Restoring checkpoint from " << checkpoint_file
                  << std::endl;
        alps::hdf5::archive cp(checkpoint_file, "r");
        cp[checkpoint_path] >> sim;
    }

    sim.run(alps::stop_callback(size_t(parameters["timelimit"])));

    // Checkpoint the simulation
    std::cout << "Checkpointing simulation to " << checkpoint_file
              << std::endl;
    alps::hdf5::archive cp(checkpoint_file, "w");
    cp[checkpoint_path] << sim;

    // Print results
    alps::results_type<sim_base>::type results = alps::collect_results(sim);
    {
        using alps::accumulators::result_wrapper;
        // std::cout << "All measured results:" << std::endl;
        // std::cout << results << std::endl;
        std::cout << sim.hamiltonian().lattice().size() << '\t';
        auto const& ppoint = sim.hamiltonian().phase_space_point();
        std::copy(ppoint.begin(), ppoint.end(),
                  std::ostream_iterator<double>{std::cout, "\t"});
        std::cout << results["Energy"].mean<double>() << '\t'
                  << results["Energy"].error<double>() << '\t'
                  << results["Magnetization"].mean<double>() << '\t'
                  << results["Magnetization"].error<double>() << '\n';

        // Saving to the output file
        std::string output_file = parameters["outputfile"];
        alps::hdf5::archive ar(output_file, "w");
        ar["/parameters"] << parameters;
        ar["/simulation/results"] << results;
    }
}
