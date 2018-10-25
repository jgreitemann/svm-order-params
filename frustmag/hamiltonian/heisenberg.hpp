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

#include "concepts.hpp"
#include "phase_space_point.hpp"
#include "site/spin_O3.hpp"
#include "update/single_flip.hpp"

#include <alps/params.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <numeric>
#include <random>
#include <utility>

namespace hamiltonian {

template <template <typename> typename Lattice>
struct heisenberg {
    using phase_point = phase_space::point::temperature;
    using site_state_type = site::spin_O3;
    using lattice_type = Lattice<site_state_type>;
    using site_iterator = typename lattice_type::iterator;

    static void define_parameters(alps::params & parameters) {
        phase_point::define_parameters(parameters, "hamiltonian.heisenberg.");
        lattice_type::define_parameters(parameters);
    }

    template <typename RNG>
    // requires RandomCreatable<site_state_type, RNG>
    heisenberg(alps::params const& parameters, RNG & rng)
        : ppoint{parameters, "hamiltonian.heisenberg."}
        , beta(1. / abs(ppoint.temp))
        , sign(ppoint.temp < 0 ? -1 : 1)
        , lattice_{parameters, [&rng] {
            return site_state_type::random(rng);
        }}
        , current_energy(total_energy())
    {
    }

    double energy() const {
        return current_energy;
    }

    double energy_per_site() const {
        return energy() / lattice().size();
    }

    double magnetization() const {
        auto sum = std::accumulate(lattice().begin(), lattice().end(),
                                   Eigen::Vector3d(Eigen::Vector3d::Zero()));
        return sum.norm() / lattice().size();
    }

    phase_point const& phase_space_point() const {
        return ppoint;
    }

    lattice_type const& lattice() const {
        return lattice_;
    }

    lattice_type & lattice() {
        return lattice_;
    }

    template <typename RNG>
    // requires UniformRandomBitGenerator<RNG>
    bool metropolis(update::single_flip_proposal<lattice_type> const& p,
                    RNG & rng)
    {
        auto nn = lattice().nearest_neighbors(p.site_it);
        auto end = lattice().end();
        Eigen::Vector3d vdiff = p.flipped - *(p.site_it);
        Eigen::Vector3d sum_nn = Eigen::Vector3d::Zero();
        for (auto it_n : nn)
            if (it_n != end)
                sum_nn += *it_n;
        double ediff = sign * vdiff.dot(sum_nn);
        if (ediff <= 0 || std::bernoulli_distribution{exp(-beta * ediff)}(rng)) {
            *(p.site_it) = std::move(p.flipped);
            current_energy += ediff;
            return true;
        }
        return false;
    }

private:
    double total_energy() const {
        // TODO implement with bonds
        double sum = 0;
        auto end = lattice().end();
        for (auto site_it = lattice().begin(); site_it != end; ++site_it)
            for (auto it_n : lattice().nearest_neighbors(site_it))
                if (it_n != end)
                    sum += site_it->dot(*it_n);
        return .5 * sign * sum;
    }

    phase_point ppoint;
    double beta;
    double sign;
    lattice_type lattice_;
    double current_energy;
};

}
