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

#ifdef USE_CONCEPTS

#include "std_concepts.hpp"

#include <alps/params.hpp>

#include <random>
#include <type_traits>
#include <utility>

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
