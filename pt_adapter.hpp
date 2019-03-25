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

#include <algorithm>
#include <chrono>
#include <iterator>
#include <map>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <alps/accumulators.hpp>
#include <alps/params.hpp>
#include <alps/mc/mcbase.hpp>

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

template <class PhasePoint>
struct pt_adapter : public alps::mcbase {
    using Base = alps::mcbase;
    using phase_point = PhasePoint;

    using batcher = iso_batcher<PhasePoint>;

    static void define_parameters(parameters_type & parameters) {
        Base::define_parameters(parameters);
        batcher::define_parameters(parameters);
    }

    pt_adapter(parameters_type & params,
               size_t seed_offset = 0)
        : Base(params, seed_offset)
    {
    }

    void rebind_communicator(mpi::communicator const& comm_new) {
        communicator = comm_new;
    }

    size_t number_of_points() const {
        return communicator.size();
    }

    bool run(boost::function<bool ()> const & stop_callback) {
        std::thread manager;
        if (communicator.rank() == 0)
            manager = std::thread(manage, communicator);
        bool ret = Base::run(stop_callback);

        // shutdown
        int my_int_status;
        bool unregistered = false;
        do {
            mpi::send(communicator, static_cast<int>(query_type::deregister), 0,
                query_tag);
            mpi::receive(communicator, my_int_status, 0, response_tag);
            unregistered = static_cast<status>(my_int_status)
                == status::unregistered;
            if (!unregistered)
                this->update();
        } while (!unregistered);

        if (communicator.rank() == 0)
            manager.join();
        return ret;
    }

    virtual phase_point phase_space_point() const {
        return slice_it->first;
    }

    virtual bool update_phase_point(phase_point const& pp) {
        using acc_ptr = std::shared_ptr<alps::accumulators::accumulator_wrapper>;
        if (slice_it->first == pp) {
            return false;
        } else {
            auto it_bool = slice_measurements.emplace(pp,
                observable_collection_type{});
            if (it_bool.second) {
                // new phase_point pp visited for the first time
                for (auto const& pair : measurements())
                    it_bool.first->second.insert(pair.first,
                        acc_ptr{pair.second->new_clone()});
                it_bool.first->second.reset();
            }
            size_t n = std::accumulate(measurements().begin(),
                measurements().end(), 0ul,
                [](size_t total, auto const& pair) {
                    return std::max(total, pair.second->count());
                });
            if (n == 0)
                slice_measurements.erase(slice_it);
            slice_it = it_bool.first;
            return true;
        }
    }

    virtual void reset_sweeps(bool skip_therm = false) {
        auto it = slice_measurements.begin();
        while (it != slice_measurements.end()) {
            if (it == slice_it)
                it++->second.reset();
            else
                slice_measurements.erase(it++);
        }
    }

    result_names_type result_names() const {
        result_names_type names;
        std::transform(measurements().begin(), measurements().end(),
            std::back_inserter(names),
            [](auto const& pair) { return pair.first; });
        return names;
    }

    results_type collect_results() const {
        return collect_results(result_names());
    }

    results_type collect_results(result_names_type const & names) const {
        results_type partial_results;
        for (auto const& name : names) {
            auto merged = measurements()[name];
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
    using observable_collection_type = typename Base::observable_collection_type;
    mpi::communicator communicator;
private:
    using slice_map_type = std::map<phase_point, observable_collection_type>;
    slice_map_type slice_measurements = {
        {{}, {}}
    };
    typename slice_map_type::iterator slice_it = slice_measurements.begin();
protected:
    observable_collection_type & measurements() {
        return slice_it->second;
    }
    observable_collection_type const& measurements() const {
        return slice_it->second;
    }

private:
    enum pt_tags {
        query_tag = 287436,
        response_tag,
        partner_tag,
        chosen_partner_tag,
        point_tag,
        weight_tag,
        acceptance_tag
    };

    enum struct status {
        unregistered,
        available,
        primary,
        secondary
    };

    enum struct query_type {
        status,
        init,
        deregister
    };

    static void manage(mpi::communicator const& comm) {
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

public:
    template <typename RNG, typename LogWeightFunc>
    bool negotiate_update(
        RNG & rng,
        bool initiate_update_if_possible,
        LogWeightFunc const& log_weight)
    {
        mpi::send(communicator, static_cast<int>(query_type::status), 0,
            query_tag);

        int response;
        mpi::receive(communicator, response, 0, response_tag);

        switch (static_cast<status>(response)) {
        case status::available:
            if (initiate_update_if_possible) {
                mpi::send(communicator, static_cast<int>(query_type::init), 0,
                    query_tag);

                int partner_rank;
                mpi::receive(communicator, partner_rank, 0, partner_tag);
                if (partner_rank == -1) {
                    // both neighbors are available, decide for one
                    partner_rank = std::bernoulli_distribution{}(rng)
                        ? (communicator.rank() - 1) : (communicator.rank() + 1);
                    mpi::send(communicator, partner_rank, 0,
                        chosen_partner_tag);
                } else if (partner_rank == -2) {
                    // neither neighbor is available, reject
                    return false;
                }

                phase_point this_point = phase_space_point();
                phase_point other_point;
                mpi::send(communicator, this_point.begin(), this_point.end(),
                    partner_rank, point_tag);
                mpi::receive(communicator, other_point.begin(),
                    other_point.end(), partner_rank, point_tag);

                double this_weight = log_weight(other_point);
                double other_weight;
                mpi::receive(communicator, other_weight, partner_rank,
                    weight_tag);

                bool acc = [&] {
                    double weight = this_weight + other_weight;
                    if (weight >= 0.
                        || std::bernoulli_distribution{exp(weight)}(rng))
                    {
                        update_phase_point(other_point);
                        return true;
                    }
                    return false;
                }();
                mpi::send(communicator, static_cast<int>(acc), partner_rank,
                    acceptance_tag);
                return acc;
            }
            break;
        case status::secondary:
            {
                int partner_rank;
                mpi::receive(communicator, partner_rank, 0, partner_tag);

                phase_point this_point = phase_space_point();
                phase_point other_point;
                mpi::receive(communicator, other_point.begin(),
                    other_point.end(), partner_rank, point_tag);
                mpi::send(communicator, this_point.begin(), this_point.end(),
                    partner_rank, point_tag);

                double this_weight = log_weight(other_point);
                mpi::send(communicator, this_weight, partner_rank, weight_tag);

                int acc;
                mpi::receive(communicator, acc, partner_rank, acceptance_tag);
                if (acc)
                    update_phase_point(other_point);
                return acc;
            }
            break;
        default:
            throw std::runtime_error("illegal response in this context");
        }
        return false;
    }
};

