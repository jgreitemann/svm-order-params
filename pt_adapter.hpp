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
#include <functional>
#include <iterator>
#include <map>
#include <random>
#include <sstream>
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
            manager = std::thread(manage, communicator, std::ref(perm));
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
        if (communicator.rank() == 0) {
            perm.resize(communicator.size());
            std::iota(perm.begin(), perm.end(), 0ul);
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
        int my_rank = communicator.rank();
        int size = communicator.size();
        for (auto const& name : names) {
            for (int rank = 0; rank < size; ++rank) {
                phase_point pt = phase_space_point();
                mpi::broadcast(communicator, pt.begin(), pt.end(), rank);
                auto it = slice_measurements.find(pt);
                int found = (it != slice_measurements.end())
                    && it->second[name].count() > 0;
                auto comm_found = mpi::split_communicator(communicator, found,
                    rank != my_rank);
                if (found) {
                    auto merged = it->second[name];
                    merged.collective_merge(comm_found, 0);
                    if (rank == my_rank)
                        partial_results.insert(name, merged.result());
                }
            }
        }
        return partial_results;
    }

    virtual void save (alps::hdf5::archive & ar) const override {
        Base::save(ar);
        auto gen_path = [](size_t i) {
            std::stringstream ss;
            ss << "pt/slice_measurements/" << i++;
            return ss.str();
        };
        size_t i = 0;
        for (auto it = slice_measurements.begin();
             it != slice_measurements.end(); ++it)
        {
            size_t n = std::accumulate(it->second.begin(),
                it->second.end(), 0ul,
                [](size_t total, auto const& pair) {
                    return std::max(total, pair.second->count());
                });
            if (n > 0) {
                auto path = gen_path(i);
                ar[path + "/point"]
                    << std::vector<double>{it->first.begin(), it->first.end()};
                ar[path + "/measurements"] << it->second;
                if (it == typename slice_map_type::const_iterator{slice_it})
                    ar["pt/slice_measurements/index"] << i;
                ++i;
            }
        }
        ar["pt/slice_measurements/size"] << i;
        if (communicator.rank() == 0)
            ar["pt/permutation"] << perm;
    }

    virtual void load (alps::hdf5::archive & ar) override {
        Base::load(ar);

        size_t n, index;
        if (communicator.rank() == 0 && communicator.size() > 1)
            ar["pt/permutation"] >> perm;
        ar["pt/slice_measurements/size"] >> n;
        ar["pt/slice_measurements/index"] >> index;

        auto gen_path = [](size_t i) {
            std::stringstream ss;
            ss << "pt/slice_measurements/" << i++;
            return ss.str();
        };

        // old measurements prototype
        using acc_ptr = std::shared_ptr<alps::accumulators::accumulator_wrapper>;
        auto & prototype_measurements = measurements();
        prototype_measurements.reset();
        auto old_slice_measurements
            = std::exchange(slice_measurements, slice_map_type{});

        for (size_t i = 0; i < n; ++i) {
            auto path = gen_path(i);
            std::vector<double> pt_vec;
            ar[path + "/point"] >> pt_vec;
            auto it_bool = slice_measurements.emplace(std::piecewise_construct,
                std::forward_as_tuple(pt_vec.begin()),
                std::forward_as_tuple());
            if (!it_bool.second)
                throw std::runtime_error(
                    "Duplicate point in slice_measurements in pt_adapter::load");
            ar[path + "/measurements"] >> it_bool.first->second;

            // reinstate empty accumulators based on prototype
            auto p_it = prototype_measurements.begin();
            auto p_end = prototype_measurements.end();
            auto m_it = it_bool.first->second.begin();
            auto m_end = it_bool.first->second.end();
            for (; p_it != p_end; ++p_it, ++m_it)
                for (; p_it != p_end && (m_it == m_end || p_it->first != m_it->first); ++p_it)
                    it_bool.first->second.insert(p_it->first,
                        acc_ptr{p_it->second->new_clone()});

            if (i == index)
                slice_it = it_bool.first;
        }
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
    std::vector<size_t> perm;

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
        acceptance,
        rejection,
        deregister
    };

    static void manage(mpi::communicator const& comm, std::vector<size_t> & perm)
    {
        std::vector<status> statuses(comm.size(), status::available);
        std::vector<int> partners(comm.size());
        size_t n_registered = statuses.size();

        // invert permutation
        std::vector<int> perm_inv(perm.size());
        for (size_t i = 0; i < perm.size(); ++i)
            perm_inv[perm[i]] = static_cast<int>(i);

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
                if (statuses[rank] == status::secondary)
                    mpi::send(comm, partners[rank], rank, partner_tag);
                break;
            case query_type::init:
                if (statuses[rank] == status::available) {
                    int adjacent_ranks[2] = {
                        perm[rank] > 0 ? perm_inv[perm[rank] - 1] : -1,
                        perm[rank] < perm.size() - 1 ? perm_inv[perm[rank] + 1] : -1
                    };
                    for (int & ar : adjacent_ranks)
                        if (ar >= 0 && statuses[ar] != status::available)
                            ar = -1;
                    mpi::send(comm, std::begin(adjacent_ranks),
                        std::end(adjacent_ranks), rank, partner_tag);

                    int partner_rank;
                    if (adjacent_ranks[0] >= 0 && adjacent_ranks[1] >= 0)
                        mpi::receive(comm, partner_rank, rank, chosen_partner_tag);
                    else if (adjacent_ranks[0] >= 0)
                        partner_rank = adjacent_ranks[0];
                    else
                        partner_rank = adjacent_ranks[1];
                    if (partner_rank >= 0) {
                        partners[rank] = partner_rank;
                        partners[partner_rank] = rank;
                        statuses[rank] = status::primary;
                        statuses[partner_rank] = status::secondary;
                    }
                } else {
                    int adjacent_ranks[2] = {-1, -1};
                    mpi::send(comm, std::begin(adjacent_ranks),
                        std::end(adjacent_ranks), rank, partner_tag);
                }
                break;
            case query_type::acceptance:
                std::swap(perm[rank], perm[partners[rank]]);
                std::swap(perm_inv[perm[rank]], perm_inv[perm[partners[rank]]]);
                [[fallthrough]];
            case query_type::rejection:
                statuses[rank] = status::available;
                statuses[partners[rank]] = status::available;
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

                int adjacent_ranks[2];
                int partner_rank;
                mpi::receive(communicator, std::begin(adjacent_ranks),
                    std::end(adjacent_ranks), 0, partner_tag);
                if (adjacent_ranks[0] >= 0 && adjacent_ranks[1] >= 0) {
                    partner_rank
                        = adjacent_ranks[std::bernoulli_distribution{}(rng)];
                    mpi::send(communicator, partner_rank, 0, chosen_partner_tag);
                } else if (adjacent_ranks[0] >= 0) {
                    partner_rank = adjacent_ranks[0];
                } else {
                    partner_rank = adjacent_ranks[1];
                }
                if (partner_rank < 0)
                    return false;

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
                if (acc) {
                    // std::cout << "successful PT update (ranks " << communicator.rank()
                    // << " and " << partner_rank << ")" << std::endl;
                    mpi::send(communicator,
                        static_cast<int>(query_type::acceptance), 0, query_tag);
                } else {
                    // std::cout << "failed PT update (ranks " << communicator.rank()
                    // << " and " << partner_rank << ")" << std::endl;
                    mpi::send(communicator,
                        static_cast<int>(query_type::rejection), 0, query_tag);
                }
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

