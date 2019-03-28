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
#include "site/spin_O3.hpp"
#include "update/single_flip.hpp"
#include "update/overrelaxation.hpp"
#include "update/global_trafo.hpp"
#include "update/parallel_tempering.hpp"

#include <alps/params.hpp>
#include <alps/hdf5.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
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

    void phase_space_point(phase_point const& pp) {
        ppoint = pp;
        beta = 1. / abs(ppoint.temp);
        sign = ppoint.temp < 0 ? -1 : 1;
        current_energy = total_energy();
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

    template <typename RNG>
    // requires UniformRandomBitGenerator<RNG>
    bool metropolis(update::overrelaxation_proposal<lattice_type> const& p,
                    RNG &)
    {
        auto nn = lattice().nearest_neighbors(p.site_it);
        auto n = std::accumulate(nn.begin(), nn.end(),
                                 Eigen::Vector3d{Eigen::Vector3d::Zero()},
                                 [] (auto const& a, auto const& b) {
                                     return a + *b;
                                 });
        *(p.site_it) = site_state_type{2. * p.site_it->dot(n) / n.squaredNorm() * n
                                       - *(p.site_it)};
        return true;
    }

    template <typename RNG>
    // requires UniformRandomBitGenerator<RNG>
    bool metropolis(update::global_trafo_proposal, RNG & rng) {
        struct angle {
            double cos;
            double sin;
        };
        std::array<angle, 3> euler;
        std::generate(euler.begin(), euler.end(), [&rng] () -> angle {
                double a = std::uniform_real_distribution<double>{0, 2. * M_PI}(rng);
                return {cos(a), sin(a)};
            });
        Eigen::Matrix3d R;
        R << euler[0].cos * euler[2].cos - euler[0].sin * euler[1].sin * euler[2].sin,
            -euler[0].sin * euler[1].cos,
            -euler[0].cos * euler[2].sin - euler[0].sin * euler[1].sin * euler[2].cos,
            euler[0].cos * euler[1].sin * euler[2].sin + euler[0].sin * euler[2].cos,
            euler[0].cos * euler[1].cos,
            euler[0].cos * euler[1].sin * euler[2].cos - euler[0].sin * euler[2].sin,
            euler[1].cos * euler[2].sin,
            -euler[1].sin,
            euler[1].cos * euler[2].cos;
        if (std::bernoulli_distribution{}(rng)) {
            R.col(0) *= -1;
        }
        for (site_state_type & s : lattice()) {
            s = site_state_type{R * s};
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
