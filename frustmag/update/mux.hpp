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

// forward declaration
namespace mpi {
    struct communicator;
}

namespace update {

namespace {
    template <typename...>
    using void_t = void;

    template <typename B>
    struct negation : std::integral_constant<bool, !bool(B::value)> {};

    template <typename T>
    static auto test_rebind_communicator(int)
        -> detail::sfinae_true<decltype(
            std::declval<T>().rebind_communicator(
                std::declval<mpi::communicator const&>()))>;

    template <typename>
    static auto test_rebind_communicator(long) -> std::false_type;

    template <typename T>
    struct communicator_rebindable : decltype(test_rebind_communicator<T>(0)) {};

    template <bool...>
    struct bool_pack;

    template <bool... bs>
    using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

    template <bool... bs>
    using all_false = std::is_same<bool_pack<bs..., false>, bool_pack<false, bs...>>;

    template <bool... bs>
    using any_true = negation<all_false<bs...>>;

    template <bool... bs>
    using any_false = negation<all_true<bs...>>;
}

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

    mux(alps::params const& parameters)
        : updates{(static_cast<void_t<Updates<LatticeH>>>(0), parameters)...}
    {
    }

    template <typename RNG>
    acceptance_type update(LatticeH & hamiltonian, RNG & rng) {
        using Indices = std::make_index_sequence<sizeof...(Updates)>;
        return update_impl(hamiltonian, rng, Indices{});
    }

    template <typename = std::enable_if_t<any_true<communicator_rebindable<Updates<LatticeH>>::value...>::value>>
    void rebind_communicator(mpi::communicator const& comm_new) {
        using Indices = std::make_index_sequence<sizeof...(Updates)>;
        rebind_communicator_impl(comm_new, Indices{});
    }

private:
    template <typename RNG, size_t... I>
    acceptance_type update_impl(LatticeH & hamiltonian, RNG & rng,
                                std::index_sequence<I...>)
    {
        return {std::get<I>(updates).update(hamiltonian, rng)[0]...};
    }

    template <size_t... I>
    void rebind_communicator_impl(mpi::communicator const& comm_new,
                                  std::index_sequence<I...>)
    {
        int dummy[] =
            {rebind_communicator_if_possible(std::get<I>(updates), comm_new)...};
    }

    template <typename Update,
              typename = std::enable_if_t<communicator_rebindable<Update>::value>>
    int rebind_communicator_if_possible(Update & u,
                                        mpi::communicator const& comm_new)
    {
        u.rebind_communicator(comm_new);
        return 0;
    }

    template <typename Update,
              typename = std::enable_if_t<!communicator_rebindable<Update>::value>,
              int dummy = 0>
    int rebind_communicator_if_possible(Update &, mpi::communicator const&) {
        return 0;
    }
};

template <template <typename> typename... Updates>
struct muxer {
    template <typename LatticeH>
    using type = mux<LatticeH, Updates...>;
};

}
