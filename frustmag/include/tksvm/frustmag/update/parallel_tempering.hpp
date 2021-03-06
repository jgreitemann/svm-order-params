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

#include <array>
#include <functional>
#include <limits>
#include <random>

#include <alps/params.hpp>

#include <tksvm/sim_adapters/pt_adapter.hpp>

#include <tksvm/frustmag/concepts.hpp>


namespace tksvm {
namespace frustmag {
namespace update {

template <typename Hamil>
struct parallel_tempering {
#ifdef USE_CONCEPTS
    static_assert(Hamiltonian<Hamil>, "Hamil is not a Hamiltonian");
#endif
public:
    using hamiltonian_type = Hamil;
    using acceptance_type = std::array<double, 1>;
    using phase_point = typename hamiltonian_type::phase_point;

    struct proposal_type {
        phase_point other;
        double other_log_weight;
    };

    static void define_parameters(alps::params & parameters) {
        parameters
        .define<size_t>("pt.query_freq", 1,
            "number of PT queries in one PT update cycle")
        .define<size_t>("pt.update_sweeps", std::numeric_limits<size_t>::max(),
            "number of MC updates before PT update is initiated")
        ;
    }

    template <typename Simulation, typename... Args>
    parallel_tempering(alps::params const& parameters,
                       Simulation & sim,
                       Args &&...)
        : query_sweeps{std::max(1ul, parameters["pt.update_sweeps"].as<size_t>()
            / parameters["pt.query_freq"].as<size_t>())}
        , update_sweeps{query_sweeps * parameters["pt.query_freq"].as<size_t>()}
        , pta{static_cast<pt_adapter<phase_point> &>(sim)}
    {
    }

    template <typename RNG>
    acceptance_type update(Hamil & hamiltonian, RNG & rng) {
        ++sweep_counter;

        bool acc = (sweep_counter % query_sweeps == 0)
        && pta.negotiate_update(rng,
            sweep_counter % update_sweeps == 0,
            [&](phase_point const& other_point) {
                return hamiltonian.log_weight(other_point);
            });

        return {static_cast<double>(acc)};
    }
private:
    size_t query_sweeps, update_sweeps;
    size_t sweep_counter = 0;
    pt_adapter<phase_point> & pta;
};

}
}
}
