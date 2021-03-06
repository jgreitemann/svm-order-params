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

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <alps/params.hpp>
#include <alps/hdf5.hpp>

#include <tksvm/config/serializer.hpp>

#include <tksvm/frustmag/concepts.hpp>
#include <tksvm/frustmag/lattice/serializer.hpp>


namespace tksvm {
namespace frustmag {
namespace lattice {

template <typename Site, size_t dim, size_t N_BASIS>
struct bravais {
    static const size_t n_basis = N_BASIS;

    using value_type = Site;
    using reference = Site&;
    using const_reference = Site const&;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;
    using lengths_type = std::array<size_t, dim + 1>;
    using unitcell_type = std::array<Site, n_basis>;
    using unitcell_collection_type = std::vector<unitcell_type>;

    template <typename BaseIt>
    struct basic_unitcell_iterator;
    using unitcell_iterator =
        basic_unitcell_iterator<typename unitcell_collection_type::iterator>;
    using unitcell_const_iterator =
        basic_unitcell_iterator<typename unitcell_collection_type::const_iterator>;

    template <typename BaseIt>
    struct basic_unitcell_iterator {
        using value_type = typename BaseIt::value_type;
        using difference_type = typename BaseIt::difference_type;
        using reference = typename BaseIt::reference;
        using pointer = typename BaseIt::pointer;
        using iterator_category = std::random_access_iterator_tag;
        basic_unitcell_iterator& operator+=(difference_type n) {
            index += n;
            buit += n;
            return *this;
        }
        basic_unitcell_iterator& operator-=(difference_type n) { return *this += -n; }
        basic_unitcell_iterator& operator++() { return *this += 1; }
        basic_unitcell_iterator& operator--() { return *this += -1; }
        basic_unitcell_iterator operator++(int) {
            basic_unitcell_iterator old(*this);
            ++(*this);
            return old;
        }
        basic_unitcell_iterator operator--(int) {
            basic_unitcell_iterator old(*this);
            --(*this);
            return old;
        }
        reference operator*() const { return *buit; }
        reference operator[](difference_type n) const { return buit[n]; }
        pointer operator->() const { return buit.operator->(); }
        bool is_end() const {
            return static_cast<size_t>(index) == plengths[dim];
        }
        friend basic_unitcell_iterator operator+(basic_unitcell_iterator lhs,
                                                 difference_type n)
        {
            return lhs += n;
        }
        friend basic_unitcell_iterator operator-(basic_unitcell_iterator lhs,
                                                 difference_type n)
        {
            return lhs -= n;
        }
        friend size_t operator-(basic_unitcell_iterator const& lhs,
                                basic_unitcell_iterator const& rhs)
        {
            return lhs.buit - rhs.buit;
        }
        friend bool operator==(basic_unitcell_iterator const& lhs,
                               basic_unitcell_iterator const& rhs)
        {
            return lhs.buit == rhs.buit;
        }
        friend bool operator!=(basic_unitcell_iterator const& lhs,
                               basic_unitcell_iterator const& rhs)
        {
            return !(lhs == rhs);
        }
        basic_unitcell_iterator& move_up(size_t i) {
            if (is_end())
                return *this;
            size_t lower = plengths[i];
            size_t upper = plengths[i+1];
            *this += lower;
            if ((index % upper) / lower == 0) {
                if (periodic)
                    return *this -= upper;
                else
                    return *this += plengths[dim] - index;
            }
            return *this;
        }
        basic_unitcell_iterator up(size_t i) const {
            return basic_unitcell_iterator(*this).move_up(i);
        }
        basic_unitcell_iterator& move_down(size_t i) {
            if (is_end())
                return *this;
            size_t lower = plengths[i];
            size_t upper = plengths[i+1];
            if ((index % upper) / lower == 0) {
                if (periodic)
                    return *this += upper - lower;
                else
                    return *this += plengths[dim] - index;
            }
            return *this -= lower;
        }
        basic_unitcell_iterator down(size_t i) const {
            return basic_unitcell_iterator(*this).move_down(i);
        }
        friend struct bravais;
        basic_unitcell_iterator() : buit{}, index{}, plengths{}, periodic{} {}
        basic_unitcell_iterator(unitcell_iterator const& it)
            : basic_unitcell_iterator(it.buit, it.index, it.plengths, it.periodic)
        {
        }
    private:
        basic_unitcell_iterator(BaseIt buit,
                                difference_type index,
                                size_t const * plengths,
                                bool periodic)
            : buit(buit), index(index), plengths(plengths), periodic(periodic)
        {
        }
        BaseIt buit;
        difference_type index;
        size_t const * plengths;
        bool periodic;
    };

    template <typename UIt>
    struct basic_iterator;
    using iterator = basic_iterator<unitcell_iterator>;
    using const_iterator = basic_iterator<unitcell_const_iterator>;

    template <typename UIt>
    struct basic_iterator {
        using value_type = typename UIt::value_type::value_type;
        using difference_type = typename UIt::value_type::difference_type;
        using arr_it = decltype(std::declval<typename UIt::reference>().begin());
        using reference = typename std::iterator_traits<arr_it>::reference;
        using pointer = typename std::iterator_traits<arr_it>::pointer;
        using iterator_category = std::random_access_iterator_tag;
        basic_iterator& operator+=(difference_type n) {
            bidx += n;
            difference_type shift = bidx / n_basis;
            if (bidx < 0)
                --shift;
            uit += shift;
            bidx -= shift * n_basis;
            return *this;
        }
        basic_iterator& operator-=(difference_type n) { return *this += -n; }
        basic_iterator& operator++() { return *this += 1; }
        basic_iterator& operator--() { return *this += -1; }
        basic_iterator operator++(int) {
            basic_iterator old(*this);
            ++(*this);
            return old;
        }
        basic_iterator operator--(int) {
            basic_iterator old(*this);
            --(*this);
            return old;
        }
        reference operator*() const { return (*uit)[bidx]; }
        reference operator[](difference_type n) const {
            return *(*this + n);
        }
        pointer operator->() const {
            return uit->begin() + bidx;
        }
        friend basic_iterator operator+(basic_iterator lhs,
                                        difference_type n)
        {
            return lhs += n;
        }
        friend basic_iterator operator-(basic_iterator lhs,
                                        difference_type n)
        {
            return lhs -= n;
        }
        friend size_t operator-(basic_iterator const& lhs,
                                basic_iterator const& rhs)
        {
            return (lhs.uit - rhs.uit) * n_basis
                + (lhs.bidx - rhs.bidx);
        }
        friend bool operator==(basic_iterator const& lhs,
                               basic_iterator const& rhs)
        {
            return lhs.uit == rhs.uit && lhs.bidx == rhs.bidx;
        }
        friend bool operator!=(basic_iterator const& lhs,
                               basic_iterator const& rhs)
        {
            return !(lhs == rhs);
        }
        UIt const& cell_it() const {
            return uit;
        }
        difference_type basis_index() const {
            return bidx;
        }
        friend struct bravais;
        basic_iterator() : uit{}, bidx{} {}
        basic_iterator(iterator const& it)
            : basic_iterator(it.uit, it.bidx)
        {
        }
        basic_iterator(UIt uit, difference_type bidx)
            : uit(uit), bidx(uit.is_end() ? 0 : bidx)
        {
        }
    private:
        UIt uit;
        difference_type bidx;
    };

    static void define_parameters(alps::params & parameters) {
        parameters
            .define<size_t>("lattice.bravais.length",
                            "linear size of the bravais lattice")
            .define<bool>("lattice.bravais.periodic", 1, "PBC = true, OBC = false");
    }

    bravais() : periodic{}, cells_{}, plengths{} {}

    template <typename Generator>
    bravais(size_t L, bool p, Generator && gen)
        : periodic(p)
    {
        plengths.front() = 1;
        std::fill(plengths.begin() + 1, plengths.end(), L);
        std::partial_sum(plengths.begin(), plengths.end(), plengths.begin(),
                         std::multiplies<>{});
        cells_.reserve(plengths.back());
        std::generate_n(std::back_inserter(cells_), plengths.back(),
                        [&] {
                            using Indices = std::make_index_sequence<n_basis>;
                            return generate_cell_impl(gen, Indices{});
                        });
    }

    template <typename Generator>
    bravais(alps::params const& parameters, Generator && gen)
        : bravais(parameters["lattice.bravais.length"],
                  parameters["lattice.bravais.periodic"],
                  std::forward<Generator>(gen))
    {
    }

    size_type size() const {
        return cells_.size() * n_basis;
    }

    iterator begin() {
        return {{cells_.begin(), 0, plengths.data(), periodic}, 0};
    }

    const_iterator begin() const {
        return {{cells_.begin(), 0, plengths.data(), periodic}, 0};
    }

    const_iterator cbegin() const {
        return {{cells_.begin(), 0, plengths.data(), periodic}, 0};
    }

    iterator end() {
        return {{cells_.end(), static_cast<difference_type>(plengths.back()),
                    plengths.data(), periodic}, 0};
    }

    const_iterator end() const {
        return {{cells_.end(), static_cast<difference_type>(plengths.back()),
                    plengths.data(), periodic}, 0};
    }

    const_iterator cend() const {
        return {{cells_.end(), static_cast<difference_type>(plengths.back()),
                    plengths.data(), periodic}, 0};
    }

    unitcell_collection_type const& cells() const {
        return cells_;
    }

    size_type max_size() const {
        return cells_.max_size();
    }

    bool empty() const {
        return cells_.empty();
    }

    void swap(bravais & other) {
        std::swap(periodic, other.periodic);
        cells_.swap(other.cells_);
        std::swap(plengths, other.plengths);
    }

    friend bool operator==(bravais const& lhs, bravais const& rhs) {
        return lhs.periodic == rhs.periodic
            && lhs.cells_ == rhs.cells_;
    }

    friend bool operator!=(bravais const& lhs, bravais const& rhs) {
        return !(lhs == rhs);
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
    template <typename Generator, size_t... I>
    unitcell_type generate_cell_impl(Generator && gen,
                                     std::index_sequence<I...>) const
    {
        return {((void)I, gen())...};
    }

    bool periodic;
    unitcell_collection_type cells_;
    lengths_type plengths;
};

}
}

namespace config {

template <typename Site, size_t dim, size_t N_BASIS>
struct serializer<frustmag::lattice::bravais<Site, dim, N_BASIS>>
    : frustmag::lattice::serializer<frustmag::lattice::bravais<Site, dim, N_BASIS>> {};

}
}
