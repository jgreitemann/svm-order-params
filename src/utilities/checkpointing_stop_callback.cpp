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

#include "checkpointing_stop_callback.hpp"

#include <alps/utilities/signal.hpp>
#include <alps/utilities/boost_mpi.hpp>

checkpointing_stop_callback::checkpointing_stop_callback(
    std::size_t timelimit,
    std::size_t checkpointtime,
    const std::function<void ()> action)
    : limit(std::chrono::seconds(timelimit)),
      cp_limit(std::chrono::seconds(checkpointtime)),
      start(Clock::now()),
      last_cp(start),
      cp_action(action)
{}

checkpointing_stop_callback::checkpointing_stop_callback(
    alps::mpi::communicator const & cm,
    std::size_t timelimit,
    std::size_t checkpointtime,
    const std::function<void ()> action)
    : limit(std::chrono::seconds(timelimit)),
      cp_limit(std::chrono::seconds(checkpointtime)),
      start(Clock::now()),
      last_cp(start),
      comm(cm),
      cp_action(action)
{}

bool checkpointing_stop_callback::operator()() {
    TimePoint now = Clock::now();
    if (now > last_cp + cp_limit) {
        cp_action();
        last_cp = now;
    }
    if (comm) {
        bool to_stop;
        if (comm->rank() == 0)
            to_stop = (!signals.empty()
                       || (limit.count() > 0
                           && now > start + limit));
        broadcast(*comm, to_stop, 0);
        return to_stop;
    } else {
        return (!signals.empty()
                || (limit.count() > 0
                    && now > start + limit));
    }
}
