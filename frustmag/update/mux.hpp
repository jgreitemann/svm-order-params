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

namespace {
    template <typename...>
    using void_t = void;

    template <typename B>
    struct negation : std::integral_constant<bool, !bool(B::value)> {};

    template <typename T, typename PTA>
    static auto test_set_pta(int)
        -> detail::sfinae_true<decltype(
            std::declval<T&>().set_pta(std::declval<PTA const&>()))>;

    template <typename, typename>
    static auto test_set_pta(long) -> std::false_type;

    template <typename T, typename PTA>
    struct is_pt_update : decltype(test_set_pta<T, PTA>(0)) {};

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

    template <typename PTA,
              typename = std::enable_if_t<any_true<is_pt_update<Updates<LatticeH>, PTA>::value...>::value>>
    void set_pta(PTA & pta) {
        using Indices = std::make_index_sequence<sizeof...(Updates)>;
        set_pta_impl(pta, Indices{});
    }

private:
    template <typename RNG, size_t... I>
    acceptance_type update_impl(LatticeH & hamiltonian, RNG & rng,
                                std::index_sequence<I...>)
    {
        return {std::get<I>(updates).update(hamiltonian, rng)[0]...};
    }

    template <typename PTA, size_t... I>
    void set_pta_impl(PTA & pta, std::index_sequence<I...>) {
        using expand = int[];
        expand{(set_pta_if_possible(std::get<I>(updates), pta), 0)...};
    }

    template <typename Update, typename PTA,
              typename = std::enable_if_t<is_pt_update<Update, PTA>::value>>
    static void set_pta_if_possible(Update & update, PTA & pta) {
        update.set_pta(pta);
    }

    template <typename Update, typename PTA,
              typename = std::enable_if_t<!is_pt_update<Update, PTA>::value>,
              int dummy = 0>
    static void set_pta_if_possible(Update &, PTA &) {}
};

template <template <typename> typename... Updates>
struct muxer {
    template <typename LatticeH>
    using type = mux<LatticeH, Updates...>;
};

}
