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
#include "config_serialization.hpp"
#include "lattice_serialization.hpp"

#include <alps/params.hpp>
#include <alps/hdf5.hpp>

#include <algorithm>
#include <array>
#include <iterator>
#include <type_traits>
#include <vector>

namespace lattice {

template <typename Site>
struct chain : std::vector<Site> {
    using Base = std::vector<Site>;
    using value_type = Site;
    using iterator = typename Base::iterator;
    using const_iterator = typename Base::const_iterator;

    static const size_t coordination = 2;
    static const size_t n_basis = 1;

    using std::vector<Site>::vector;

    static void define_parameters(alps::params & parameters) {
        parameters
            .define<size_t>("lattice.chain.length", "length of the chain")
            .define<bool>("lattice.chain.periodic", 1, "PBC = true, OBC = false");
    }

    template <typename Generator>
    chain(size_t L, bool p, Generator && gen)
        : periodic(p)
    {
        this->reserve(L);
        std::generate_n(std::back_inserter(*this), L, gen);
    }

    template <typename Generator>
    chain(alps::params const& parameters, Generator && gen)
        : chain(parameters["lattice.chain.length"],
                parameters["lattice.chain.periodic"],
                std::forward<Generator>(gen))
    {
    }

    auto nearest_neighbors(iterator it) -> std::array<iterator, 2> {
        if (periodic) {
            return {
                it == this->begin() ? std::prev(this->end()) : std::prev(it),
                ++it == this->end() ? this->begin() : it
            };
        } else {
            return {
                it == this->begin() ? this->end() : std::prev(it),
                ++it
            };
        }
    }

    auto nearest_neighbors(const_iterator it) const
        -> std::array<const_iterator, 2>
    {
        if (periodic) {
            return {
                it == this->begin() ? std::prev(this->end()) : std::prev(it),
                ++it == this->end() ? this->begin() : it
            };
        } else {
            return {
                it == this->begin() ? this->end() : std::prev(it),
                ++it
            };
        }
    }

    template <typename ..., typename T = value_type,
              typename = std::enable_if_t<is_serializable<T>::value>>
    void save(alps::hdf5::archive & ar) const {
        std::vector<double> data;
        auto it = std::back_inserter(data);
        for (auto const& site : *this)
            site.serialize(it);
        ar["data"] << data;
        ar["periodic"] << periodic;
    }

    template <typename ..., typename T = value_type,
              typename = std::enable_if_t<is_serializable<T>::value>>
    void load(alps::hdf5::archive & ar) {
        std::vector<double> data;
        ar["data"] >> data;
        auto it = data.begin();
        for (auto & site : *this)
            site.deserialize(it);
        ar["periodic"] >> periodic;
    }

private:
    bool periodic;
};

}

template <typename Site>
struct config_serializer<lattice::chain<Site>>
    : lattice_serializer<lattice::chain<Site>> {};