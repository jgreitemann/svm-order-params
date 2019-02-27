// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2019  Jonas Greitemann, Ke Liu, and Lode Pollet

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
#include "config_sim_base.hpp"
#include "mpi.hpp"
#include "phase_space_policy.hpp"
#include "svm-wrapper.hpp"
#include "test_adapter.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/mpiadapter.hpp>
#include <alps/mc/stop_callback.hpp>

#ifdef CONFIG_MAPPING_LAZY
#include "procrastination_adapter.hpp"
using sim_type = procrastination_adapter<sim_base>;
#else
#include "training_adapter.hpp"
using sim_type = training_adapter<sim_base>;
#endif

using kernel_t = typename sim_type::kernel_t;
using classifier_t = typename sim_type::phase_classifier;
using phase_point = typename classifier_t::point_type;

constexpr int report_idle_tag = 42;
constexpr int request_batch_tag = 43;

int main(int argc, char** argv)
{
    mpi::environment env(argc, argv, mpi::environment::threading::multiple);
    alps::mpi::communicator comm_world;

    const int is_master = (comm_world.rank() == 0);
    try {
        std::cout << "Initializing parameters..." << std::endl;
        alps::params parameters(argc, argv, comm_world);

        // define parameters
        sim_type::define_parameters(parameters);
        test_adapter<sim_base>::define_test_parameters(parameters);
        if (!parameters.is_restored()) {
            parameters.define<double>("nu", 0.5, "nu_SVC regularization parameter");
            parameters.define<size_t>("progress_interval", 3,
                                      "time in sec between progress reports");
        }

        if (parameters.help_requested(std::cout) ||
            parameters.has_missing(std::cout)) {
            return 1;
        }

        std::string checkpoint_file = parameters["checkpoint"].as<std::string>();

        int n_clones = comm_world.size();
        if (is_master && parameters.is_restored()) {
            int n_clones_saved;
            alps::hdf5::archive cp(checkpoint_file, "r");

            cp["simulation/n_clones"] >> n_clones_saved;
            if (n_clones != n_clones_saved) {
                throw std::runtime_error(
                    "MPI world size must match saved number of clones");
            }
        }

        // Collect phase points
        std::vector<std::vector<phase_point>> batches;
        {
            auto sweep_pol = phase_space::sweep::from_parameters<phase_point>(parameters);
            std::cout << "sweep size: " << sweep_pol->size() << std::endl;
            std::mt19937 rng{parameters["SEED"].as<size_t>()};
            phase_point pp;
            sweep_pol->yield(pp, rng);
            std::generate_n(std::back_inserter(batches), sweep_pol->size(),
                [&]() -> std::vector<phase_point> {
                    sweep_pol->yield(pp, rng);
                    return {pp};
                });
        }

        size_t batch_size = 1;
        size_t n_group = comm_world.size() / batch_size;

        std::cout << "n_group: " << n_group << std::endl;

        std::thread dispatcher = [&] {
            if (is_master) {
                return std::thread{[&] {
                    for (size_t batch_index = 0; batch_index < batches.size(); ++batch_index) {
                        std::cout << "waiting for idle process\n";
                        int idle = mpi::receive(comm_world, MPI_ANY_SOURCE, report_idle_tag);
                        std::cout << "process " << idle << " is idle\n";
                        mpi::send(comm_world, static_cast<int>(batch_index), idle, request_batch_tag);
                    }
                    for (size_t group = 0; group < n_group; ++group) {
                        std::cout << "waiting for idle process\n";
                        int idle = mpi::receive(comm_world, MPI_ANY_SOURCE, report_idle_tag);
                        std::cout << "process " << idle << " will be told to terminate\n";
                        mpi::send(comm_world, -1, idle, request_batch_tag);
                    }
                }};
            } else {
                return std::thread{};
            }
        }();

        int batch_index;
        auto request_batch = [&] {
            mpi::send(comm_world, 0, report_idle_tag);
            mpi::receive(comm_world, batch_index, 0, request_batch_tag);
            return batch_index >= 0;
        };

        std::mt19937 rng{static_cast<size_t>(comm_world.rank())};
        while (request_batch()) {
            std::cout << "process " << comm_world.rank()
                      << " begins work on batch " << batch_index << '\n';
            std::this_thread::sleep_for(std::chrono::milliseconds(std::uniform_int_distribution<size_t>{3000, 4000}(rng)));
            std::cout << "process " << comm_world.rank()
                      << " finished work on batch " << batch_index << '\n';
        }

        std::cout << "process " << comm_world.rank() << " terminates.\n";

        if (is_master)
            dispatcher.join();

        return 0;
    } catch (const std::runtime_error& exc) {
        std::cout << "Exception caught: " << exc.what() << std::endl;
        return 2;
    } catch (...) {
        std::cout << "Unknown exception caught." << std::endl;
        return 2;
    }
}
