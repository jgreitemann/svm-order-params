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

#include <hamiltonian/heisenberg.hpp>
#include <lattice/kagome.hpp>

#include <alps/accumulators.hpp>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

namespace obs {
    using measurements_t = alps::accumulators::accumulator_set;

    template <typename Obs>
    measurements_t & operator<<(measurements_t & meas, Obs const& o) {
        return o.measure(meas);
    }

    template <typename LatticeH>
    struct energy {
        LatticeH const& hamiltonian;

        static measurements_t & define(measurements_t & meas) {
            return meas
                << alps::accumulators::FullBinningAccumulator<double>("Energy")
                << alps::accumulators::FullBinningAccumulator<double>("Energy^2");
        }

        static std::vector<std::string> names() {
            return {"Energy"};
        }

        measurements_t & measure(measurements_t & meas) const {
            double E = hamiltonian.energy_per_site();
            meas["Energy"] << E;
            meas["Energy^2"] << E * E;
            return meas;
        }
    };

    template <typename LatticeH>
    struct magnetization {
        LatticeH const& hamiltonian;

        static measurements_t & define(measurements_t & meas) {
            return meas
                << alps::accumulators::FullBinningAccumulator<double>("|Magnetization|")
                << alps::accumulators::FullBinningAccumulator<double>("Magnetization")
                << alps::accumulators::FullBinningAccumulator<double>("Magnetization^2");
        }

        static std::vector<std::string> names() {
            return {"|Magnetization|", "Magnetization"};
        }

        measurements_t & measure(measurements_t & meas) const {
            double mag = hamiltonian.magnetization();
            meas["|Magnetization|"] << abs(mag);
            meas["Magnetization"] << mag;
            meas["Magnetization^2"] << mag * mag;
            return meas;
        }
    };

    template <typename LatticeH>
    struct nematicity {
        LatticeH const& hamiltonian;

        static measurements_t & define(measurements_t & meas) {
            return meas
                << alps::accumulators::FullBinningAccumulator<double>("Nematicity")
                << alps::accumulators::FullBinningAccumulator<double>("Nematicity^2");
        }

        static std::vector<std::string> names() {
            return {"Nematicity"};
        }

        measurements_t & measure(measurements_t & meas) const {
            using site_t = typename LatticeH::site_state_type;
            double Q[site_t::size * site_t::size] {};
            for (auto const& Si : hamiltonian.lattice()) {
                for (size_t a = 0, i = 0; a < site_t::size; ++a)
                    for (size_t b = 0; b < site_t::size; ++b, ++i)
                        Q[i] += Si[a] * Si[b];
            }
            double nem = std::inner_product(std::begin(Q), std::end(Q),
                                            std::begin(Q), 0)
                / pow(hamiltonian.lattice().size(), 2) - 1./3;
            nem = std::max(nem, 0.);
            meas["Nematicity"] << sqrt(nem);
            meas["Nematicity^2"] << nem;
            return meas;
        }
    };

    template <typename LatticeH>
    struct octupolar {
        LatticeH const& hamiltonian;

        static measurements_t & define(measurements_t & meas) {
            return meas
                << alps::accumulators::FullBinningAccumulator<double>("Octupolarity")
                << alps::accumulators::FullBinningAccumulator<double>("Octupolarity^2");
        }

        static std::vector<std::string> names() {
            return {"Octupolarity"};
        }

        measurements_t & measure(measurements_t & meas) const {
            using site_t = typename LatticeH::site_state_type;
            double T[site_t::size * site_t::size * site_t::size] {};
            double S[site_t::size] {};
            for (auto const& Si : hamiltonian.lattice()) {
                for (size_t a = 0, i = 0; a < site_t::size; ++a)
                    for (size_t b = 0; b < site_t::size; ++b)
                        for (size_t c = 0; c < site_t::size; ++c, ++i)
                            T[i] += Si[a] * Si[b] * Si[c];
                for (size_t a = 0; a < site_t::size; ++a)
                    S[a] += Si[a];
            }
            double nem = (std::inner_product(std::begin(T), std::end(T),
                                             std::begin(T), 0)
                          - 3./5 * std::inner_product(std::begin(S), std::end(S),
                                                      std::begin(S), 0))
                / pow(hamiltonian.lattice().size(), 2);
            nem = std::max(nem, 0.);
            meas["Octupolarity"] << sqrt(nem);
            meas["Octupolarity^2"] << nem;
            return meas;
        }
    };

    template <typename LatticeH>
    struct constraint {
        LatticeH const& hamiltonian;

        static measurements_t & define(measurements_t & meas) {
            return meas << alps::accumulators::FullBinningAccumulator<double>("Gamma")
                        << alps::accumulators::FullBinningAccumulator<double>("Gamma^2");
        }

        static std::vector<std::string> names() {
            return {"Gamma"};
        }

        measurements_t & measure(measurements_t & meas) const {
            using site_t = typename LatticeH::site_state_type;
            double sum_norm = 0.;
            for (auto const& cell : hamiltonian.lattice().cells()) {
                double sum[site_t::size] = {};
                for (auto const& Sj : cell) {
                    for (size_t i = 0; i < site_t::size; ++i) {
                        sum[i] += Sj[i];
                    }
                }
                sum_norm += std::inner_product(std::begin(sum), std::end(sum),
                                               std::begin(sum), 0.);
            }
            sum_norm = 1 - sum_norm / hamiltonian.lattice().size();
            meas["Gamma"] << sum_norm;
            meas["Gamma^2"] << pow(sum_norm, 2);
            return meas;
        }
    };

    template <typename LatticeH, template <typename> typename... Obs>
    struct mux : Obs<LatticeH>... {
        mux(LatticeH const& hamiltonian) : Obs<LatticeH>{hamiltonian}... {}

        static measurements_t & define(measurements_t & meas) {
            // more elegantly with fold expressions (C++17)
            // return (Obs<LatticeH>::define(meas), ...);

            // hack -- cf. https://stackoverflow.com/a/25683817/2788450
            [[maybe_unused]] int dummy[] =
                {0, (static_cast<void>(Obs<LatticeH>::define(meas)), 0)...};
            return meas;
        }

        static std::vector<std::string> names() {
            return [] (std::initializer_list<std::vector<std::string>> il) {
                std::vector<std::string> ret;
                ret.reserve(std::accumulate(il.begin(), il.end(), 0,
                                            [] (size_t l, std::vector<std::string> const& v) {
                                                return l + v.size();
                                            }));
                for (std::vector<std::string> const& v : il)
                    std::copy(v.begin(), v.end(), std::back_inserter(ret));
                return ret;
            }({Obs<LatticeH>::names()...});
        }

        measurements_t & measure(measurements_t & meas) const {
            // more elegantly with fold expressions (C++17)
            // return (meas << ... << static_cast<Obs<LatticeH> const&>(*this));

            [[maybe_unused]] int dummy[] =
                {0, (static_cast<void>(
                         meas << static_cast<Obs<LatticeH> const&>(*this)
                         ), 0)...};
            return meas;
        }
    };
}

template <typename LatticeH>
struct observables : obs::mux<LatticeH, obs::energy, obs::magnetization> {
    observables(LatticeH const& hamiltonian)
        : obs::mux<LatticeH, obs::energy, obs::magnetization>{hamiltonian} {}
};

template <>
struct observables<hamiltonian::heisenberg<lattice::kagome>>
    : obs::mux<hamiltonian::heisenberg<lattice::kagome>,
               obs::energy, obs::magnetization, obs::nematicity, obs::octupolar,
               obs::constraint>
{
    observables(hamiltonian::heisenberg<lattice::kagome> const& hamiltonian)
        : obs::mux<hamiltonian::heisenberg<lattice::kagome>,
                   obs::energy, obs::magnetization, obs::nematicity,
                   obs::octupolar, obs::constraint>{hamiltonian} {}
};
