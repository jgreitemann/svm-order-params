// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2017  Jonas Greitemann, Ke Liu, and Lode Pollet

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

#pragma once

#include <chrono>
#include <functional>

#include <alps/config.hpp>
#include <alps/utilities/signal.hpp>
#include <alps/utilities/mpi.hpp>

#include <boost/optional.hpp>

class checkpointing_stop_callback {
public:
    checkpointing_stop_callback(std::size_t timelimit,
                                std::size_t checkpointtime,
                                const std::function<void ()> action);
    checkpointing_stop_callback(alps::mpi::communicator const & cm,
                                std::size_t timelimit,
                                std::size_t checkpointtime,
                                const std::function<void ()> action);
    bool operator()();
private:
    typedef std::chrono::high_resolution_clock Clock;
    typedef typename Clock::duration Duration;
    typedef typename Clock::time_point TimePoint;

    const Duration limit;
    const Duration cp_limit;
    alps::signal signals;
    const TimePoint start;
    TimePoint last_cp;
    boost::optional<alps::mpi::communicator> comm;
    const std::function<void ()> cp_action;
};
