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


template <class Simulation>
struct embarrassing_adapter : public alps::mcmpiadapter<Simulation> {
    using Base = alps::mcmpiadapter<Simulation>;
    using parameters_type = typename Simulation::parameters_type;
    using phase_point = typename Simulation::phase_point;

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
        mpi::communicator comm)
        : Base(params, comm)
    {
    }

    void rebind_communicator(mpi::communicator const& comm_new) {
        communicator = comm_new;
    }

    size_t number_of_points() const {
        return 1ul;
    }

protected:
    using Base::communicator;
};
