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
#include <vector>

#include <alps/mc/mcbase.hpp>
#include <alps/mc/mpiadapter.hpp>

#include <boost/function.hpp>

template <typename PhasePoint>
struct embarrassing_adapter : public alps::mcmpiadapter<alps::mcbase> {
    using Base = alps::mcmpiadapter<alps::mcbase>;
    using phase_point = PhasePoint;

    struct batcher {
        using batch_type = std::vector<phase_point>;
        using batches_type = std::vector<batch_type>;

        static void define_parameters(parameters_type & parameters) {
            parameters.template define<size_t>("batch.n_parallel", 0,
                "number of parallel processes per batch (0 = optimum)");
        }

        batcher(parameters_type & parameters)
            : n_parallel{parameters["batch.n_parallel"].template as<size_t>()}
        {
        }

        template <typename Container>
        batches_type operator()(Container const& points) const {
            size_t np = n_parallel ? n_parallel
                : std::max<size_t>(mpi::communicator{}.size() / points.size(), 1);
            batches_type batches;
            std::transform(points.begin(), points.end(),
                std::back_inserter(batches),
                [np](phase_point const& pp) {
                    return batch_type(np, pp);
                });
            return batches;
        }

    private:
        size_t n_parallel;
    };

    static void define_parameters(parameters_type & parameters) {
        Base::define_parameters(parameters);
        batcher::define_parameters(parameters);
    }

    embarrassing_adapter(parameters_type & params,
                         size_t seed_offset)
        : Base(params, mpi::communicator{}, 1, seed_offset)
    {
    }

    bool run(boost::function<bool ()> const & stop_callback) {
        bool done = false, stopped = false;
        do {
            this->update();
            this->measure();
            if (stopped || schedule_checker.pending()) {
                stopped = stop_callback();
                double local_fraction = stopped ? 1. : this->fraction_completed();
                schedule_checker.update(
                    fraction = mpi::all_reduce(communicator, local_fraction,
                        std::plus<double>()));
                done = fraction >= 1.;
            }
        } while(!done);
        return !stopped;
    }

    void rebind_communicator(mpi::communicator const& comm_new) {
        communicator = comm_new;
    }

    size_t number_of_points() const {
        return 1ul;
    }

    virtual void reset_sweeps(bool skip_therm) = 0;
    virtual phase_point phase_space_point() const = 0;
    virtual bool update_phase_point(phase_point const&) = 0;

protected:
    using Base::communicator;

    using observable_collection_type = typename Base::observable_collection_type;
    observable_collection_type & measurements() {
        return Base::measurements;
    }
    observable_collection_type const& measurements() const {
        return Base::measurements;
    }
};
