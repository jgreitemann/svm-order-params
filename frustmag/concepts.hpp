// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2018  Jonas Greitemann, Ke Liu, and Lode Pollet

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

#include <alps/params.hpp>

#include <iterator>
#include <random>
#include <type_traits>
#include <utility>

namespace detail {
    template <typename T>
    struct sfinae_true : std::true_type {};

    template <typename T>
    static auto test_load(int)
        -> sfinae_true<decltype(std::declval<T>().load(std::declval<alps::hdf5::archive&>()))>;

    template <typename>
    static auto test_load(long) -> std::false_type;

    template <typename T, typename InputIt>
    static auto test_deserialize(int)
        -> sfinae_true<decltype(std::declval<T>().deserialize(std::declval<InputIt&>()))>;

    template <typename, typename>
    static auto test_deserialize(long) -> std::false_type;

    template <typename T>
    static auto test_save(int)
        -> sfinae_true<decltype(std::declval<T const>().save(std::declval<alps::hdf5::archive&>()))>;

    template <typename>
    static auto test_save(long) -> std::false_type;

    template <typename T, typename OutputIt>
    static auto test_serialize(int)
        -> sfinae_true<decltype(std::declval<T const>().serialize(std::declval<OutputIt&>()))>;

    template <typename, typename>
    static auto test_serialize(long) -> std::false_type;
}

template <typename T>
struct has_load : decltype(detail::test_load<T>(0)) {};

template <typename T, typename InputIt = std::istream_iterator<double>>
struct has_deserialize : decltype(detail::test_deserialize<T, InputIt>(0)) {};

template <typename T>
struct has_save : decltype(detail::test_save<T>(0)) {};

template <typename T, typename OutputIt = std::ostream_iterator<double>>
struct has_serialize : decltype(detail::test_serialize<T, OutputIt>(0)) {};

template <typename T>
struct is_archivable
    : std::integral_constant<bool, has_load<T>::value && has_save<T>::value> {};

template <typename T>
struct is_serializable
    : std::integral_constant<bool, (has_deserialize<T>::value
                                    && has_serialize<T>::value)>
{
};

#ifdef USE_CONCEPTS

#include "std_concepts.hpp"

template <typename T, UniformRandomBitGenerator RNG = std::mt19937>
concept bool RandomCreatable = requires(RNG & rng) {
    {T::random(rng)} -> T;
};

template <typename T, typename... X>
concept bool ParameterConstructible = requires(alps::params & params, X&&... x) {
    {T::define_parameters(params)} -> void;
    {T{params, std::forward<X>(x)...}} -> T;
};

template <typename T, UniformRandomBitGenerator RNG = std::mt19937>
concept bool SiteState = requires {
    requires RandomCreatable<T, RNG>;
    requires requires(T s, RNG & rng) {
        {s.flipped(rng)} -> T;
    };
};

template <typename T>
struct DummyGenerator {
    T operator()() const;
};

template <typename T, typename Generator = DummyGenerator<typename T::value_type>>
concept bool Lattice = requires {
    requires Container<T>;
    requires ParameterConstructible<T, std::add_rvalue_reference_t<Generator>>;
    {T::coordination} -> size_t;
    requires requires(T lat, typename T::iterator site_it) {
        {lat.nearest_neighbors(site_it)};
        requires Container<decltype(lat.nearest_neighbors(site_it))>;
        requires Same<typename decltype(lat.nearest_neighbors(site_it))::value_type,
                      typename T::iterator>;
    };
    requires requires(T const lat, typename T::const_iterator site_it) {
        {lat.nearest_neighbors(site_it)};
        requires Container<decltype(lat.nearest_neighbors(site_it))>;
        requires Same<typename decltype(lat.nearest_neighbors(site_it))::value_type,
                      typename T::const_iterator>;
    };
};

template <typename T, typename RNG = std::mt19937>
concept bool Hamiltonian = requires(T h) {
    typename T::phase_point;
    requires ParameterConstructible<T, std::add_lvalue_reference_t<RNG>>;
    {h.phase_space_point()} -> typename T::phase_point;
    requires requires(T h, typename T::phase_point pp) {
        {h.phase_space_point(pp)};
    }
    {h.energy()} -> double;
};

template <typename T>
concept bool MagneticHamiltonian = requires(T h) {
    {h.magnetization()} -> double;
};

template <typename T, typename RNG = std::mt19937>
concept bool LatticeHamiltonian = requires(T h) {
    typename T::lattice_type;
    typename T::site_state_type;
    requires Hamiltonian<T, RNG>;
    {h.energy_per_site()} -> double;
};

template <typename U, typename RNG = std::mt19937>
concept bool MetropolisUpdate = requires {
    typename U::hamiltonian_type;
    typename U::proposal_type;
    requires DefaultConstructible<U>;
    requires requires(U & u, typename U::hamiltonian_type & h, RNG & rng,
                      typename U::proposal_type && p)
    {
        {u.update(h, rng)} -> void;
        {h.metropolis(p, rng)} -> bool;
    };
};

#endif
