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

#include <alps/mc/mcbase.hpp>
#include <alps/mc/mpiadapter.hpp>


template <class Simulation>
class embarrassing_adapter : public alps::mcmpiadapter<Simulation> {
public:
    using Base = alps::mcmpiadapter<Simulation>;
    using parameters_type = typename Simulation::parameters_type;
    // using results_type = typename Simulation::results_type;
    // using result_names_type = typename Simulation::result_names_type;

    static void define_parameters(alps::params & parameters) {
        Base::define_parameters(parameters);
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
    using Simulation::measurements;
    using Base::communicator;
};
