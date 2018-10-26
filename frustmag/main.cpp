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

#include "frustmag.hpp"

#include <alps/mc/stop_callback.hpp>

#include <algorithm>
#include <iterator>
#include <iostream>

#include "hamiltonian/heisenberg.hpp"
#include "hamiltonian/ising.hpp"
#include "lattice/chain.hpp"
#include "lattice/ortho.hpp"
#include "lattice/triangular.hpp"
#include "update/single_flip.hpp"

#if defined HEISENBERG
template <template <typename> typename Lattice>
using hamiltonian_t_t = hamiltonian::heisenberg<Lattice>;
#elif defined ISING
template <template <typename> typename Lattice>
using hamiltonian_t_t = hamiltonian::ising<Lattice>;
#else
#error Unknown hamiltonian
#endif

#if defined CHAIN
using hamiltonian_t = hamiltonian_t_t<lattice::chain>;
#elif defined SQUARE
using hamiltonian_t = hamiltonian_t_t<lattice::square>;
#elif defined CUBIC
using hamiltonian_t = hamiltonian_t_t<lattice::cubic>;
#elif defined TRIANGULAR
using hamiltonian_t = hamiltonian_t_t<lattice::triangular>;
#else
#error Unknown lattice
#endif

#ifdef USE_CONCEPTS
namespace {
    template <typename T>
    requires Lattice<T>
    struct check_lattice {};
    template struct check_lattice<typename hamiltonian_t::lattice_type>;

    template <typename T>
    requires Hamiltonian<T>
    struct check_hamiltonian {};
    template struct check_hamiltonian<hamiltonian_t>;

    template <typename U>
    requires MetropolisUpdate<U>
    struct check_update {};
    template struct check_update<update::single_flip<hamiltonian_t>>;
}
#endif

using sim_type = frustmag_sim<hamiltonian_t, update::single_flip>;

int main(int argc, char** argv)
{
    // Creates the parameters for the simulation
    // If an hdf5 file is supplied, reads the parameters there
    std::cout << "Initializing parameters..." << std::endl;
    alps::params parameters(argc, argv);

    // define parameters
    sim_type::define_parameters(parameters);

    if (parameters.help_requested(std::cout) ||
        parameters.has_missing(std::cout)) {
        return 1;
    }

    std::string checkpoint_file = parameters["checkpoint"].as<std::string>();

    sim_type sim(parameters);

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
    alps::results_type<sim_type>::type results = alps::collect_results(sim);
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
