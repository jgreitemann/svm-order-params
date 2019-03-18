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

#include "argh.h"
#include "checkpointing_stop_callback.hpp"
#include "config_sim_base.hpp"
#include "dispatcher.hpp"
#include "embarrassing_adapter.hpp"
#include "mpi.hpp"
#include "phase_space_policy.hpp"
#include "svm-wrapper.hpp"
#include "test_adapter.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>

#ifdef CONFIG_MAPPING_LAZY
#include "procrastination_adapter.hpp"
using sim_type = embarrassing_adapter<procrastination_adapter<sim_base>>;
#else
#include "training_adapter.hpp"
using sim_type = embarrassing_adapter<training_adapter<sim_base>>;
#endif

using kernel_t = typename sim_type::kernel_t;
using classifier_t = typename sim_type::phase_classifier;
using phase_point = typename classifier_t::point_type;


int main(int argc, char** argv)
{
    mpi::environment env(argc, argv, mpi::environment::threading::multiple);
    mpi::communicator comm_world;

    const bool is_master = (comm_world.rank() == 0);
    try {
        if (is_master)
            std::cout << "Initializing parameters..." << std::endl;
        alps::params parameters(argc, argv, comm_world);
        argh::parser cmdl(argc, argv);

        // define parameters
        sim_type::define_parameters(parameters);
        test_adapter<sim_base>::define_test_parameters(parameters);
        if (!parameters.is_restored()) {
            parameters.define<double>("nu", 0.5, "nu_SVC regularization parameter");
        }

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
        using batches_type = std::vector<std::vector<phase_point>>;
        auto get_batches = [&] {
            batches_type batches;
            auto sweep_pol = phase_space::sweep::from_parameters<phase_point>(parameters);
            std::mt19937 rng{parameters["SEED"].as<size_t>()};
            size_t repeat;
            cmdl("repeat", 1) >> repeat;
            std::generate_n(std::back_inserter(batches), sweep_pol->size(),
                [&, p=phase_point{}]() mutable -> std::vector<phase_point> {
                    sweep_pol->yield(p, rng);
                    return std::vector<phase_point>(repeat, p);
                });
            return batches;
        };

        sim_type sim(parameters, comm_world);

        const std::string checkpoint_file = parameters["checkpoint"];
        const bool resumed = parameters.is_restored();
        alps::stop_callback stop_cb(parameters["timelimit"].as<size_t>());

        using proxy_t = dispatcher<batches_type>::archive_proxy_type;

        mpi::mutex archive_mutex(comm_world);

        dispatcher<batches_type> dispatch(checkpoint_file,
            archive_mutex,
            resumed,
            get_batches(),
            stop_cb,
            [&](proxy_t ar) { log() << "restoring checkpoint\n"; ar >> sim; },
            [&](proxy_t ar) { log() << "writing checkpoint\n"; ar << sim; });

        while (dispatch.request_batch()) {
            bool valid = dispatch.valid();
            auto comm_valid = mpi::split_communicator(dispatch.comm_group, valid);
            if (!valid)
                continue;
            sim.rebind_communicator(comm_valid);
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
            sim.run(stop_cb);
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
