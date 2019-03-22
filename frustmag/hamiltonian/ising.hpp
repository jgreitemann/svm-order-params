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
#include "phase_space_point.hpp"
#include "site/spin_Z2.hpp"
#include "update/single_flip.hpp"
#include "update/global_trafo.hpp"
#include "update/parallel_tempering.hpp"

#include <alps/params.hpp>
#include <alps/hdf5.hpp>

#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace hamiltonian {

template <template <typename> typename Lattice>
struct ising {
    using phase_point = phase_space::point::temperature;
    using site_state_type = site::spin_Z2;
    using lattice_type = Lattice<site_state_type>;
    using site_iterator = typename lattice_type::iterator;

    static void define_parameters(alps::params & parameters) {
        phase_point::define_parameters(parameters, "hamiltonian.ising.");
        lattice_type::define_parameters(parameters);
    }

    template <typename RNG>
    // requires RandomCreatable<site_state_type, RNG>
    ising(alps::params const& parameters, RNG & rng)
        : ppoint{parameters, "hamiltonian.ising."}
        , sign(ppoint.temp < 0 ? -1 : 1)
        , lattice_{parameters, [&rng] {
            return site_state_type::random(rng);
        }}
        , current_int_energy(total_int_energy())
    {
        for (size_t i = 0; i <= lattice_type::coordination; ++i)
            iexp[i] = exp(-2. * i / abs(ppoint.temp));
    }

    double energy() const {
        return current_int_energy;
    }

    double energy_per_site() const {
        return energy() / lattice().size();
    }

    double magnetization() const {
        double sum = std::accumulate(lattice().begin(), lattice().end(), 0);
        return sum / lattice().size();
    }

    lattice_type const& lattice() const {
        return lattice_;
    }

    lattice_type & lattice() {
        return lattice_;
    }

    phase_point const& phase_space_point() const {
        return ppoint;
    }

    void phase_space_point(phase_point const& pp) {
        ppoint = pp;
        sign = ppoint.temp < 0 ? -1 : 1;
        current_int_energy = total_int_energy();
        for (size_t i = 0; i <= lattice_type::coordination; ++i)
            iexp[i] = exp(-2. * i / abs(ppoint.temp));
    }

    template <typename RNG>
    // requires UniformRandomBitGenerator<RNG>
    bool metropolis(update::single_flip_proposal<lattice_type> const& p,
                    RNG & rng)
    {
        auto nn = lattice().nearest_neighbors(p.site_it);
        auto end = lattice().end();
        int sum_nn = 0;
        for (auto it_n : nn)
            if (it_n != end)
                sum_nn += p.flipped.dot(*it_n);
        int ediff = 2 * sign * sum_nn;
        if (ediff <= 0 || std::bernoulli_distribution{iexp[ediff/2]}(rng)) {
            *(p.site_it) = std::move(p.flipped);
            current_int_energy += ediff;
            return true;
        }
        return false;
    }

    template <typename RNG>
    // requires UniformRandomBitGenerator<RNG>
    bool metropolis(update::global_trafo_proposal, RNG &) {
        for (site_state_type & s : lattice()) {
            s.s *= -1;
        }
        return true;
    }

    double log_weight(phase_point const& other) const {
        return (ppoint.temp - other.temp) * energy();
    }

    virtual void save(alps::hdf5::archive & ar) const {
        ar["phase_point"] << std::vector<double>{ppoint.begin(), ppoint.end()};
        ar["lattice"] << lattice_;
    }

    virtual void load(alps::hdf5::archive & ar) {
        ar["lattice"] >> lattice_;

        std::vector<double> pp;
        ar["phase_point"] >> pp;
        if (pp.size() != phase_point::label_dim)
            throw std::runtime_error("error reading phase point");
        phase_space_point(phase_point{pp.begin()});
    }

private:
    int total_int_energy() const {
        // TODO implement with bonds
        int sum = 0;
        auto end = lattice().end();
        for (auto site_it = lattice().begin(); site_it != end; ++site_it)
            for (auto it_n : lattice().nearest_neighbors(site_it))
                if (it_n != end)
                    sum += site_it->dot(*it_n);
        return sum * sign / 2;
    }

    phase_point ppoint;
    int sign;
    lattice_type lattice_;
    int current_int_energy;
    std::array<double, lattice_type::coordination+1> iexp;
};

}
