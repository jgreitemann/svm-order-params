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
#include <stdexcept>
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
constexpr int read_checkpoint_tag = 44;
constexpr int write_checkpoint_tag = 45;

int main(int argc, char** argv)
{
    mpi::environment env(argc, argv, mpi::environment::threading::multiple);
    mpi::communicator comm_world;

    const bool is_master = (comm_world.rank() == 0);
    try {
        if (is_master)
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

        if (parameters.help_requested(std::cout)
            || parameters.has_missing(std::cout))
        {
            return 1;
        }

        std::string checkpoint_file = parameters["checkpoint"].as<std::string>();

        const bool resumed = parameters.is_restored();

        // Collect phase points
        std::vector<std::vector<phase_point>> batches;
        {
            auto sweep_pol = phase_space::sweep::from_parameters<phase_point>(parameters);
            std::mt19937 rng{parameters["SEED"].as<size_t>()};
            phase_point pp;
            sweep_pol->yield(pp, rng);
            std::generate_n(std::back_inserter(batches), sweep_pol->size(),
                [&]() -> std::vector<phase_point> {
                    sweep_pol->yield(pp, rng);
                    return {pp};
                });
        }

        size_t batch_size = batches.front().size();
        size_t n_group = comm_world.size() / batch_size;
        int this_group = comm_world.rank() / batch_size;

        auto comm_group = mpi::split_communicator(comm_world, this_group);
        const bool is_group_leader = (comm_group.rank() == 0);

        auto log = [&](std::ostream & os = std::cout) -> std::ostream & {
            return os << '[' << comm_world.rank() << '/' << comm_world.size()
            << ';' << comm_group.rank() << '/' << comm_group.size() << "]\t";
        };

        alps::stop_callback stop_cb(parameters["timelimit"].as<size_t>());

        std::thread dispatcher = [&] {
            if (is_master) {
                return std::thread{[&] {
                    size_t batch_index = 0;
                    std::vector<size_t> active_batches;
                    std::vector<bool> to_resume_flag;
                    if (resumed) {
                        {
                            alps::hdf5::archive cp(checkpoint_file, "r");
                            cp["simulation/active_batches"] >> active_batches;
                        }
                        mpi::send(comm_world, 0, read_checkpoint_tag);
                        if (n_group != active_batches.size()) {
                            throw std::runtime_error(
                                "number of groups mustn't change on resumption");
                        }
                        to_resume_flag.resize(n_group, true);
                        batch_index = *std::max_element(active_batches.begin(),
                            active_batches.end()) + 1;
                    } else {
                        active_batches.resize(n_group);
                        to_resume_flag.resize(n_group, false);
                    }

                    // dispatch batches until exhausted or stopped
                    size_t n_cleanup = n_group;
                    while (n_cleanup > 0) {
                        int idle = mpi::receive(comm_world, MPI_ANY_SOURCE,
                            report_idle_tag);
                        if (to_resume_flag[idle]) {
                            // tell group to resume its active batch
                            mpi::send(comm_world,
                                static_cast<int>(active_batches[idle]),
                                idle, request_batch_tag);
                            to_resume_flag[idle] = false;
                        } else if (stop_cb() || batch_index >= batches.size()) {
                            mpi::send(comm_world, -1, idle, request_batch_tag);
                            --n_cleanup;
                        } else {
                            // dispatch a new batch
                            mpi::send(comm_world,
                                static_cast<int>(batch_index),
                                idle, request_batch_tag);
                            active_batches[idle] = batch_index;
                            ++batch_index;
                        }
                    }

                    // wait for the last process to checkpoint, then write
                    // the last active batches
                    mpi::receive(comm_world, comm_world.size() - 1,
                        write_checkpoint_tag);
                    alps::hdf5::archive cp(checkpoint_file, "w");
                    cp["simulation/active_batches"] << active_batches;
                }};
            } else {
                return std::thread{};
            }
        }();

        sim_type sim(parameters, comm_world.rank());

        if (resumed) {
            // Ring lock for reading checkpoints
            if (is_master)
                mpi::receive(comm_world, 0, read_checkpoint_tag);
            else
                mpi::receive(comm_world, comm_world.rank() - 1,
                    read_checkpoint_tag);

            std::string checkpoint_path = [&] {
                std::stringstream ss;
                ss << "simulation/clones/" << comm_world.rank();
                return ss.str();
            } ();
            std::cout << "Restoring simulation from " << checkpoint_file
                      << " (process " << comm_world.rank() << ")"
                      << std::endl;
            {
                alps::hdf5::archive cp(checkpoint_file, "r");
                cp[checkpoint_path] >> sim;
            }

            if (comm_world.rank() + 1 < comm_world.size())
                mpi::send(comm_world, comm_world.rank() + 1, read_checkpoint_tag);
        }

        int batch_index;
        auto request_batch = [&] {
            if (is_group_leader) {
                mpi::send(comm_world, 0, report_idle_tag);
                mpi::receive(comm_world, batch_index, 0, request_batch_tag);
            }
            mpi::broadcast(comm_group, batch_index, 0);
            return batch_index >= 0;
        };

        mpi::barrier(comm_world);

        bool freshly_restored = resumed;
        while (request_batch()) {
            size_t slice_index = comm_group.rank();
            auto slice_point = batches[batch_index][slice_index];
            log() << "working on batch " << batch_index << ": "
                  << slice_point << '\n';
            if (freshly_restored) {
                freshly_restored = false;
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
            sim.run(stop_cb);
        }

        {
            // Ring lock for writing checkpoints
            if (!is_master)
                mpi::receive(comm_world, comm_world.rank() - 1, write_checkpoint_tag);
            std::string checkpoint_path = [&] {
                std::stringstream ss;
                ss << "simulation/clones/" << comm_world.rank();
                return ss.str();
            } ();
            std::cout << "Checkpointing simulation to " << checkpoint_file
                      << " (process " << comm_world.rank() << ")"
                      << std::endl;
            {
                alps::hdf5::archive cp(checkpoint_file, "w");
                cp["simulation/n_clones"] << comm_world.size();
                cp[checkpoint_path] << sim;
            }
            mpi::send(comm_world, (comm_world.rank() + 1) % comm_world.size(),
                write_checkpoint_tag);
            // This ring lock cycles back to the beginning to signal the
            // dispatcher to write active batches.
        }

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
