// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2018-2019  Jonas Greitemann, Ke Liu, and Lode Pollet

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

#include "concepts.hpp"

#include <alps/params.hpp>

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace update {

template <typename LatticeH, template <typename> typename... Updates>
struct mux {
#ifdef USE_CONCEPTS
    static_assert(LatticeHamiltonian<LatticeH>,
                  "LatticeH is not a LatticeHamiltonian");
#endif
public:
    using hamiltonian_type = LatticeH;
    using acceptance_type = std::array<double, sizeof...(Updates)>;

    std::tuple<Updates<LatticeH>...> updates;

    static void define_parameters(alps::params & parameters) {
        using expand = int[];
        expand{(Updates<LatticeH>::define_parameters(parameters), 0)...};
    }

    template <typename... Args>
    mux(alps::params const& parameters, Args &&... args)
        : updates{Updates<LatticeH>{parameters, std::forward<Args>(args)...}...}
    {
    }

    template <typename RNG>
    acceptance_type update(LatticeH & hamiltonian, RNG & rng) {
        using Indices = std::make_index_sequence<sizeof...(Updates)>;
        return update_impl(hamiltonian, rng, Indices{});
    }

private:
    template <typename RNG, size_t... I>
    acceptance_type update_impl(LatticeH & hamiltonian, RNG & rng,
                                std::index_sequence<I...>)
    {
        return {std::get<I>(updates).update(hamiltonian, rng)[0]...};
    }
};

template <template <typename> typename... Updates>
struct muxer {
    template <typename LatticeH>
    using type = mux<LatticeH, Updates...>;
};

}
