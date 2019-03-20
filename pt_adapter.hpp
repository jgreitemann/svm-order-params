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
#include <iterator>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include <alps/accumulators.hpp>
#include <alps/params.hpp>

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
        communicator = comm_new;
    }

    size_t number_of_points() const {
        return communicator.size();
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
};
