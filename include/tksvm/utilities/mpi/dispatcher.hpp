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

#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <alps/hdf5/archive.hpp>

#include <tksvm/utilities/mpi/mpi.hpp>


namespace tksvm {
namespace mpi {

template <typename BatchesContainer>
struct dispatcher {
	using stop_callback_type = std::function<bool()>;
	using archive_proxy_type =
		alps::hdf5::detail::archive_proxy<alps::hdf5::archive>;
	using checkpoint_callback_type = std::function<void(archive_proxy_type)>;

	using batch_type = typename BatchesContainer::value_type;
	using point_type = typename batch_type::value_type;

private:
	static constexpr int report_idle_tag = 42;
	static constexpr int request_batch_tag = 43;

	mpi::communicator comm_world;

public:
	const std::string checkpoint_file;
	const bool resumed;
	size_t session_counter = 0;
	const BatchesContainer batches;
	const size_t batch_size = std::max_element(batches.begin(), batches.end(),
		[](batch_type const& lhs, batch_type const& rhs) {
			return lhs.size() < rhs.size();
		})->size();
	const size_t n_group = comm_world.size() / batch_size;
	const size_t this_group = comm_world.rank() / batch_size;
	const mpi::communicator comm_group =
		mpi::split_communicator(comm_world, this_group);
	const bool is_master = (comm_world.rank() == 0);
	const bool is_group_leader = (comm_group.rank() == 0);

private:
	mpi::mutex & archive_mutex;
	int batch_int;

	stop_callback_type stop_cb;
	checkpoint_callback_type write_cb;
	std::thread background_thread;

public:
	dispatcher(std::string const& checkpoint_file,
		mpi::mutex & archive_mutex,
		bool resumed,
		BatchesContainer && batches,
		stop_callback_type && stop_cb,
		checkpoint_callback_type const& read_cb,
		checkpoint_callback_type && write_cb)
	: checkpoint_file{checkpoint_file}
	, resumed{resumed}
	, batches{std::forward<BatchesContainer>(batches)}
    , archive_mutex{archive_mutex}
	, stop_cb{std::move(stop_cb)}
	, write_cb{std::move(write_cb)}
	{
        if (resumed) {
            std::string checkpoint_path = [&] {
                std::stringstream ss;
                ss << "simulation/clones/" << comm_world.rank();
                return ss.str();
            } ();
            {
            	std::lock_guard<mpi::mutex> archive_lock(archive_mutex);
                alps::hdf5::archive cp(checkpoint_file, "r");
                read_cb(cp[checkpoint_path]);
            }
        }

        if (is_master)
            background_thread = std::thread{[this] { dispatch_job(); }};

	}

	dispatcher(dispatcher const&) = delete;
	dispatcher& operator=(dispatcher const&) = delete;
	dispatcher(dispatcher &&) = delete;
	dispatcher& operator=(dispatcher &&) = delete;

	~dispatcher() {
        std::string checkpoint_path = [&] {
            std::stringstream ss;
            ss << "simulation/clones/" << comm_world.rank();
            return ss.str();
        } ();

        if (is_master)
        	background_thread.join();

        {
        	std::lock_guard<mpi::mutex> archive_lock(archive_mutex);
            alps::hdf5::archive cp(checkpoint_file, "w");
            cp["simulation/n_clones"] << comm_world.size();
            write_cb(cp[checkpoint_path]);
        }
	}

	bool request_batch() {
        if (is_group_leader) {
            mpi::send(comm_world, 0, report_idle_tag);
            mpi::receive(comm_world, batch_int, 0, request_batch_tag);
        }
        mpi::broadcast(comm_group, batch_int, 0);
        ++session_counter;
        return batch_int >= 0;
	}

	bool valid() const {
		return batch_int >= 0 && static_cast<size_t>(batch_int) < batches.size()
			&& static_cast<size_t>(comm_group.rank()) < batches[batch_int].size();
	}

	bool point_resumed() const {
		return resumed && session_counter <= 1;
	}

	size_t batch_index() const {
		return batch_int;
	}

	point_type point() const {
		return valid() ? batches[batch_int][comm_group.rank()] : point_type{};
	}

private:
	void dispatch_job() {
        size_t batch_index = 0;
        std::vector<size_t> active_batches;
        std::vector<bool> to_resume_flag;
        if (resumed) {
            {
            	std::lock_guard<mpi::mutex> archive_lock(archive_mutex);
                alps::hdf5::archive cp(checkpoint_file, "r");
                cp["simulation/active_batches"] >> active_batches;
            }
            if (n_group != active_batches.size()) {
                throw std::runtime_error(
                    "number of groups mustn't change on resumption");
            }
            to_resume_flag.resize(n_group + 1, true);
            to_resume_flag[n_group] = false;
            batch_index = *std::max_element(active_batches.begin(),
                active_batches.end()) + 1;
        } else {
            active_batches.resize(n_group);
            to_resume_flag.resize(n_group + 1, false);
        }

        // dispatch batches until exhausted or stopped
        size_t n_cleanup = n_group + (comm_world.size() % batch_size != 0);
        while (n_cleanup > 0) {
            int idle = mpi::spin_receive(comm_world, MPI_ANY_SOURCE,
                report_idle_tag, [&] {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(100));
                });
            size_t idle_group = idle / batch_size;
            if (to_resume_flag[idle_group]) {
                // tell group to resume its active batch
                mpi::send(comm_world,
                    static_cast<int>(active_batches[idle]),
                    idle, request_batch_tag);
                to_resume_flag[idle] = false;
            } else if (stop_cb() || batch_index >= batches.size()
            	|| idle_group == n_group)
            {
                mpi::send(comm_world, -1, idle, request_batch_tag);
                --n_cleanup;
            } else {
                // dispatch a new batch
                mpi::send(comm_world,
                    static_cast<int>(batch_index),
                    idle, request_batch_tag);
                active_batches[idle_group] = batch_index;
                ++batch_index;
            }
        }

        {
        	std::lock_guard<mpi::mutex> archive_lock(archive_mutex);
        	alps::hdf5::archive cp(checkpoint_file, "w");
        	cp["simulation/active_batches"] << active_batches;
        }
	}
};

}
}
