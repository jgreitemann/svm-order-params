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

#include "mpi.hpp"
#include "pt.hpp"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <map>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <alps/accumulators.hpp>
#include <alps/params.hpp>

#include <boost/function.hpp>

template <typename Point>
struct iso_batcher {
    using batch_type = std::vector<Point>;
    using batches_type = std::vector<batch_type>;

    static void define_parameters(alps::params & parameters) {
        parameters.template define<size_t>("batch.index", 0,
            "index of the phase space point over which PT is performed");
    }

    iso_batcher(alps::params & parameters)
        : index{parameters["batch.index"].template as<size_t>()}
    {
    }

    template <typename Container>
    batches_type operator()(Container const& points) const {
        std::map<Point, batch_type> map;
        typename std::map<Point, batch_type>::iterator it;
        for (Point const& p : points) {
            Point key = p;
            *(key.begin() + index) = 0.;
            std::tie(it, std::ignore) = map.emplace(key, batch_type{});
            it->second.push_back(p);
        }
        batches_type batches;
        std::transform(map.begin(), map.end(), std::back_inserter(batches),
            [](auto const& p) { return p.second; });
        return batches;
    }

private:
    size_t index;
};

template <class Simulation,
          typename Batcher = iso_batcher<typename Simulation::phase_point>>
struct pt_adapter : public Simulation {
    using parameters_type = typename Simulation::parameters_type;
    using phase_point = typename Simulation::phase_point;
    using results_type = typename Simulation::results_type;
    using result_names_type = typename Simulation::result_names_type;

    using batcher = Batcher;

    static void define_parameters(parameters_type & parameters) {
        Simulation::define_parameters(parameters);
        batcher::define_parameters(parameters);
    }

    pt_adapter(parameters_type & params,
        mpi::communicator comm,
        size_t seed_offset = 0)
        : Simulation(params, seed_offset * comm.size() + comm.rank())
        , communicator{comm}
    {
    }

    void rebind_communicator(mpi::communicator const& comm_new) {
        Simulation::rebind_communicator(comm_new);
        communicator = comm_new;
    }

    size_t number_of_points() const {
        return communicator.size();
    }

    bool run(boost::function<bool ()> const & stop_callback) {
        std::thread manager;
        if (communicator.rank() == 0)
            manager = std::thread(manage, communicator);
        bool ret = Simulation::run(stop_callback);

        // shutdown
        int my_int_status;
        bool unregistered = false;
        do {
            mpi::send(communicator,
                static_cast<int>(pt::query_type::deregister),
                0, pt::query_tag);
            mpi::receive(communicator, my_int_status, 0, pt::response_tag);
            unregistered = static_cast<pt::status>(my_int_status)
                == pt::status::unregistered;
            if (!unregistered)
                Simulation::update();
        } while (!unregistered);

        if (communicator.rank() == 0)
            manager.join();
        return ret;
    }

    bool update_phase_point(phase_point const& pp) {
        using acc_ptr = std::shared_ptr<alps::accumulators::accumulator_wrapper>;
        auto it_bool = slice_measurements.emplace(
            Simulation::phase_space_point(),
            observable_collection_type{});
        if (measurements.begin()->second->count() > 0) {
            if (it_bool.second)
                for (auto const& pair : measurements)
                    it_bool.first->second.insert(pair.first,
                        acc_ptr{pair.second->new_clone()});
            else
                it_bool.first->second.merge(measurements);
            measurements.reset();
        }
        return Simulation::update_phase_point(pp);
    }

    void reset_sweeps(bool skip_therm = false) {
        Simulation::reset_sweeps(skip_therm);
        slice_measurements.clear();
    }

    results_type collect_results() const {
        return collect_results(this->result_names());
    }

    results_type collect_results(result_names_type const & names) const {
        results_type partial_results;
        for (auto const& name : names) {
            auto merged = measurements[name];
            auto it = slice_measurements.find(Simulation::phase_space_point());
            if (it != slice_measurements.end())
                merged.merge(it->second[name]);
            partial_results.insert(name, merged.result());
        }
        return partial_results;
/*
        results_type partial_results;
        for (auto it = names.begin(); it != names.end(); ++it) {
            size_t has_count = (this->measurements[*it].count() > 0);
            const size_t sum_counts =
                mpi::all_reduce(communicator, has_count, std::plus<size_t>());
            if (static_cast<int>(sum_counts) == communicator.size()) {
                auto merged = this->measurements[*it];
                merged.collective_merge(communicator, 0);
                partial_results.insert(*it, merged.result());
            } else if (sum_counts > 0
                && static_cast<int>(sum_counts) < communicator.size())
            {
                throw std::runtime_error(*it
                    + " was measured on only some of the MPI processes.");
            }
        }
        return partial_results;
*/
    }

protected:
    using observable_collection_type = typename Simulation::observable_collection_type;
    using Simulation::measurements;
    mpi::communicator communicator;

private:
    std::map<phase_point, observable_collection_type> slice_measurements;

    static void manage(mpi::communicator const& comm) {
        using namespace pt;
        std::vector<status> statuses(comm.size(), status::available);
        std::vector<int> partners(comm.size());
        size_t n_registered = statuses.size();

        while (n_registered > 0) {
            int int_query_type;
            int rank = mpi::spin_receive(comm, int_query_type, MPI_ANY_SOURCE,
                query_tag, [] {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                });
            switch (static_cast<query_type>(int_query_type)) {
            case query_type::status:
                mpi::send(comm, static_cast<int>(statuses[rank]), rank,
                    response_tag);
                if (statuses[rank] == status::secondary) {
                    mpi::send(comm, partners[rank], rank, partner_tag);
                    statuses[rank] = status::available;
                    statuses[partners[rank]] = status::available;
                }
                break;
            case query_type::init:
                if (statuses[rank] == status::available) {
                    int partner_rank = -2;
                    bool lower = rank > 0
                        && statuses[rank - 1] == status::available;
                    bool upper = rank < static_cast<int>(statuses.size()) - 1
                        && statuses[rank + 1] == status::available;
                    if (lower && upper) {
                        mpi::send(comm, -1, rank, partner_tag);
                        mpi::receive(comm, partner_rank, rank,
                            chosen_partner_tag);
                    } else if (lower) {
                        partner_rank = rank - 1;
                        mpi::send(comm, partner_rank, rank, partner_tag);
                    } else if (upper) {
                        partner_rank = rank + 1;
                        mpi::send(comm, partner_rank, rank, partner_tag);
                    } else {
                        mpi::send(comm, -2, rank, partner_tag);
                    }
                    if (partner_rank >= 0) {
                        partners[rank] = partner_rank;
                        partners[partner_rank] = rank;
                        statuses[rank] = status::primary;
                        statuses[partner_rank] = status::secondary;
                    }
                } else {
                    mpi::send(comm, -2, rank, partner_tag);
                }
                break;
            case query_type::deregister:
                if (statuses[rank] == status::available) {
                    statuses[rank] = status::unregistered;
                    --n_registered;
                }
                mpi::send(comm, static_cast<int>(statuses[rank]), rank,
                    response_tag);
                break;
            default:
                throw std::runtime_error("Invalid PT query type");
                break;
            }
        }
    }
};

