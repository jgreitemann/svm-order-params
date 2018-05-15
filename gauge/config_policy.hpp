// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2017  Jonas Greitemann, Ke Liu, and Lode Pollet

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

#include "combinatorics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include <boost/multi_array.hpp>

#include <Eigen/Dense>


using indices_t = std::vector<size_t>;

template <char first_letter>
struct basic_indices_t {
    indices_t ind;
    size_t & operator[] (size_t i) { return ind[i]; }
    size_t const& operator[] (size_t i) const { return ind[i]; }
    friend std::ostream & operator << (std::ostream & os,
                                       basic_indices_t const& indices)
    {
        for (size_t i : indices.ind)
            os << static_cast<char>(first_letter + i);
        return os;
    }
};

using block_indices_t = basic_indices_t<'l'>;
using contraction_indices_t = basic_indices_t<'a'>;
using component_indices_t = basic_indices_t<'x'>;

namespace element_policy {

    struct uniaxial {
        static const size_t n_color = 1;
        static const size_t n_block = 1;
        static const size_t range = 3 * n_color;
        inline size_t sublattice(size_t index) const { return 0; }
        inline size_t color(size_t index) const { return 2; }
        inline size_t component(size_t index) const { return index; }

        size_t rearranged_index (indices_t const& ind) const {
            size_t components = 0;
            for (auto it = ind.begin(); it != ind.end(); ++it) {
                components *= 3;
                components += component(*it);
            }
            return components;
        }
    };

    struct triad {
        static const size_t n_color = 3;
        static const size_t n_block = 3;
        static const size_t range = 3 * n_color;
        inline size_t sublattice(size_t index) const { return 0; }
        inline size_t color(size_t index) const { return index / 3; }
        inline size_t component(size_t index) const { return index % 3; }

        size_t rearranged_index (indices_t const& ind) const {
            size_t components = 0;
            size_t colors = 0;
            size_t shift = 1;
            for (auto it = ind.begin(); it != ind.end(); ++it) {
                components *= 3;
                components += component(*it);
                colors *= 3;
                colors += color(*it);
                shift *= 3;
            }
            return colors * shift + components;
        }
    };

    template <typename BaseElementPolicy, size_t N_UNIT>
    struct n_partite : private BaseElementPolicy {
        static const size_t n_unitcell = N_UNIT;
        static const size_t n_color = BaseElementPolicy::n_color;
        static const size_t n_block = n_unitcell * BaseElementPolicy::n_color;
        static const size_t range = n_unitcell * BaseElementPolicy::range;
        inline size_t sublattice(size_t index) const {
            return index / BaseElementPolicy::range;
        }
        inline size_t color(size_t index) const {
            return BaseElementPolicy::color(index % BaseElementPolicy::range);
        }
        inline size_t component(size_t index) const {
            return BaseElementPolicy::component(index % BaseElementPolicy::range);
        }

        size_t rearranged_index (indices_t const& ind) const {
            size_t sublats = 0;
            size_t shift = 1;
            for (auto it = ind.begin(); it != ind.end(); ++it) {
                sublats *= n_unitcell;
                sublats += sublattice(*it);
                shift *= n_unitcell;
            }
            indices_t base_ind(ind);
            for (size_t & i : base_ind)
                i = i % BaseElementPolicy::range;
            return sublats + BaseElementPolicy::rearranged_index(base_ind) * shift;
        }
    };

    template <typename BaseElementPolicy>
    using bipartite = n_partite<BaseElementPolicy, 2>;
}

namespace lattice {

    template <typename BaseElementPolicy, typename Container>
    struct uniform {
        using ElementPolicy = BaseElementPolicy;
        using site_const_iterator = typename Container::const_iterator;

        struct unitcell;
        struct const_iterator {
            const_iterator & operator++ () { ++sit; return *this; }
            const_iterator operator++ (int) {
                const_iterator old(*this);
                ++(*this);
                return old;
            }
            const_iterator & operator-- () { --sit; return *this; }
            const_iterator operator-- (int) {
                const_iterator old(*this);
                --(*this);
                return old;
            }
            friend bool operator== (const_iterator lhs, const_iterator rhs) { return lhs.sit == rhs.sit; }
            friend bool operator!= (const_iterator lhs, const_iterator rhs) { return lhs.sit != rhs.sit; }
            unitcell operator* () const { return {sit}; }
            std::unique_ptr<unitcell> operator-> () const {
                return std::unique_ptr<unitcell>(new unitcell(sit));
            }
            friend uniform;
        private:
            const_iterator (site_const_iterator it) : sit {it} {}
            site_const_iterator sit;
        };

        struct unitcell {
            typename Container::value_type const& sublattice (size_t i = 0) const {
                if (i != 0)
                    throw std::runtime_error("invalid sublattice index");
                return it[i];
            }
            friend const_iterator;
        private:
            unitcell (site_const_iterator it) : it {it} {}
            site_const_iterator it;
        };

        uniform (Container const& linear) : linear(linear) {}

        const_iterator begin () const {
            return {linear.begin()};
        }

        const_iterator end () const {
            return {linear.end()};
        }

        size_t size () const {
            return linear.size();
        }

    private:
        Container const& linear;
    };

    template <typename BaseElementPolicy, typename Container, size_t DIM = 3>
    struct square {
        using ElementPolicy = element_policy::bipartite<BaseElementPolicy>;
        using site_const_iterator = typename Container::const_iterator;
        using coord_t = std::array<size_t, DIM>;

        struct unitcell;
        struct const_iterator {
            const_iterator & operator++ () {
                coord[0] += 2;
                if (coord[0] >= L) {
                    size_t i;
                    for (i = 1; i < coord.size(); ++i) {
                        ++coord[i];
                        if (coord[i] < L)
                            break;
                        coord[i] = 0;
                    }
                    if (i == coord.size()) {
                        coord[0] = L;
                    } else {
                        size_t sum = 0;
                        for (i = 1; i < coord.size(); ++i) {
                            sum += coord[i];
                        }
                        coord[0] = sum % 2;
                    }
                }
                return *this;
            }
            const_iterator operator++ (int) {
                const_iterator old(*this);
                ++(*this);
                return old;
            }
            friend bool operator== (const_iterator lhs, const_iterator rhs) {
                return (lhs.coord == rhs.coord
                        && lhs.root == rhs.root
                        && lhs.L == rhs.L);
            }
            friend bool operator!= (const_iterator lhs, const_iterator rhs) { return !(lhs == rhs); }
            unitcell operator* () const { return {root, lin_index(), L}; }
            std::unique_ptr<unitcell> operator-> () const {
                return std::unique_ptr<unitcell>(root, lin_index(), L);
            }
            friend square;
        private:
            const_iterator (site_const_iterator it, coord_t c, size_t L)
                : root{it}, coord{c}, L{L} {}
            size_t lin_index () const {
                size_t sum = 0;
                for (size_t c : coord) {
                    sum *= L;
                    sum += c;
                }
                return sum;
            }
            site_const_iterator root;
            coord_t coord;
            size_t L;
        };

        struct unitcell {
            typename Container::value_type const& sublattice (size_t i) const {
                if (i == 0)
                    return root[idx];
                else if (i == 1)
                    return root[idx / L * L + (idx + 1) % L];
                else
                    throw std::runtime_error("invalid sublattice index");
            }
            friend const_iterator;
        private:
            unitcell (site_const_iterator it, size_t idx, size_t L)
                : root{it}, idx{idx}, L{L} {}
            site_const_iterator root;
            size_t idx;
            size_t L;
        };

        square (Container const& linear) : linear(linear) {
            L = static_cast<size_t>(pow(linear.size() + 0.5, 1./DIM));
            if (combinatorics::ipow(L, DIM) != linear.size())
                throw std::runtime_error("linear configuration size doesn't match DIM");
            if (L % 2 != 0)
                throw std::runtime_error("lattice not bipartite w.r.t. PBCs");
        }

        const_iterator begin () const {
            return {linear.begin(), {0}, L};
        }

        const_iterator end () const {
            return {linear.begin(), {L}, L};
        }

        size_t size () const {
            return linear.size() / 2;
        }

    private:
        Container const& linear;
        size_t L;
    };

}


namespace symmetry_policy {

    struct none {
        size_t size (size_t range, size_t rank) const {
            return combinatorics::ipow(range, rank);
        }

        void advance_ind (indices_t & ind, size_t range) const {
            auto rit = ind.rbegin();
            ++(*rit);
            while (*rit == range) {
                ++rit;
                if (rit == ind.rend())
                    break;
                ++(*rit);
                auto it = rit.base();
                while (it != ind.end()) {
                    *it = 0;
                    ++it;
                }
            }
        }

        bool transform_ind (indices_t & ind) const {
            return false;
        }

        size_t number_of_equivalents (indices_t const& ind) const {
            return 1;
        }
    };

    struct symmetrized {
        size_t size (size_t range, size_t rank) const {
            return combinatorics::binomial(rank + range - 1, rank);
        }

        void advance_ind (indices_t & ind, size_t range) const {
            auto rit = ind.rbegin();
            ++(*rit);
            while (*rit == range) {
                ++rit;
                if (rit == ind.rend())
                    break;
                ++(*rit);
                auto it = rit.base();
                while (it != ind.end()) {
                    *it = *rit;
                    ++it;
                }
            }
        }

        bool transform_ind (indices_t & ind) const {
            return std::next_permutation(ind.begin(), ind.end());
        }

        size_t number_of_equivalents (indices_t const& ind) const {
            return combinatorics::number_of_permutations(ind);
        }
    };

}

namespace block_reduction {

    constexpr size_t inf = std::numeric_limits<size_t>::infinity();

    template <size_t N>
    struct norm {
        norm & operator+= (double x) {
            sum += std::pow(std::abs(x), N);
            ++M;
            return *this;
        }
        operator double () const {
            return std::pow(sum, 1./N);
        }
    private:
        double sum = 0.;
        size_t M = 0;
    };

    template <>
    struct norm<inf> {
        norm & operator+= (double x) {
            if (std::abs(x) > max)
                max = std::abs(x);
            return *this;
        }
        operator double () const {
            return max;
        }
    private:
        double max = 0.;
    };

    struct sum {
        sum & operator+= (double x) {
            sum += x;
            return *this;
        }
        operator double () const {
            return sum;
        }
    private:
        double sum = 0;
    };
}

namespace detail {
    struct contraction {
        typedef std::vector<size_t> ep_type;
        contraction (ep_type && e) : endpoints(std::forward<ep_type>(e)) {}

        bool is_self_contraction () const {
            for (auto ep : endpoints)
                if (ep < endpoints.size())
                    return true;
            return false;
        }

        bool operator() (indices_t const& i_ind, indices_t const& j_ind) const {
            std::vector<bool> checked(2 * endpoints.size(), false);
            auto ind_get = [&] (size_t i) {
                return i >= i_ind.size() ? j_ind[i - i_ind.size()] : i_ind[i];
            };
            auto it = endpoints.begin();
            for (size_t i = 0; i < 2 * endpoints.size(); ++i) {
                if (checked[i])
                    continue;
                size_t a = ind_get(i);
                size_t b = ind_get(*it);
                if (ind_get(i) != ind_get(*it))
                    return false;
                checked[*it] = true;
                ++it;
            }
            return true;
        }

        friend std::ostream & operator<< (std::ostream & os, contraction const& ct) {
            char c = 'a';
            std::string out(2 * ct.endpoints.size(), '-');
            auto it = ct.endpoints.begin();
            for (auto & o : out) {
                if (o == '-') {
                    o = c++;
                    out[*(it++)] = o;
                }
            }
            return os << '[' << out.substr(0, ct.endpoints.size())
                    << ';' << out.substr(ct.endpoints.size(), ct.endpoints.size())
                    << ']';
        }
    private:
        ep_type endpoints;
    };
}

struct config_policy {
    typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> local_state;
    typedef boost::multi_array<local_state, 1> config_array;
    typedef boost::multi_array<double, 2> matrix_t;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> contraction_matrix_t;

    virtual size_t size () const = 0;
    virtual size_t range () const = 0;
    virtual size_t rank () const = 0;
    virtual std::vector<double> configuration (config_array const&) const = 0;

    virtual matrix_t rearrange_by_component (matrix_t const& c) const = 0;
    virtual std::pair<matrix_t, matrix_t> block_structure (matrix_t const& c) const = 0;

    virtual indices_t block_indices(indices_t const& ind) const = 0;
    virtual indices_t component_indices(indices_t const& ind) const = 0;

    virtual contraction_matrix_t contraction_matrix(std::vector<detail::contraction> const& cs,
                                                    indices_t const& i_col_ind,
                                                    indices_t const& j_col_ind) const = 0;

    std::vector<detail::contraction> contractions() const {
        std::vector<detail::contraction> cs;
        std::set<size_t> places;
        for (size_t i = 0; i < 2 * rank(); ++i)
            places.insert(i);
        rec_cs(cs, {}, places);
        return cs;
    }

    virtual std::set<indices_t> all_block_indices () const = 0;

private:
    void rec_cs(std::vector<detail::contraction> & cs,
                std::vector<size_t> ep,
                std::set<size_t> pl) const {
        if (pl.size() == 2) {
            ep.push_back(*(++pl.begin()));
            cs.push_back(std::move(ep));
        } else {
            pl.erase(pl.begin());
            std::vector<size_t> epn(ep);
            for(auto it = pl.begin(); it != pl.end(); ++it) {
                epn.push_back(*it);
                pl.erase(it);
                rec_cs(cs, epn, pl);
                it = pl.insert(epn.back()).first;
                epn.pop_back();
            }
        }
    }

};

template <typename LatticePolicy, typename SymmetryPolicy,
          typename ElementPolicy = typename LatticePolicy::ElementPolicy>
struct gauge_config_policy : public config_policy, private ElementPolicy, SymmetryPolicy {

    gauge_config_policy (size_t rank, bool unsymmetrize = true)
        : rank_(rank), unsymmetrize(unsymmetrize), weights(size())
    {
        indices_t ind(rank_);
        for (double & w : weights) {
            w = sqrt(SymmetryPolicy::number_of_equivalents(ind));
            advance_ind(ind);
        }
    }

    virtual size_t size () const override {
        return SymmetryPolicy::size(ElementPolicy::range, rank_);
    }

    virtual std::vector<double> configuration (config_array const& R) const override {
        std::vector<double> v(size());
        indices_t ind(rank_);
        LatticePolicy lattice(R);
        auto w_it = weights.begin();
        for (double & elem : v) {
            for (auto cell : lattice) {
                double prod = 1;
                for (size_t a : ind)
                    prod *= cell.sublattice(sublattice(a))(color(a), component(a));
                elem += prod;
            }
            elem *= *w_it / lattice.size();

            advance_ind(ind);
            ++w_it;
        }
        return v;
    }

    virtual matrix_t rearrange_by_component (matrix_t const& c) const override {
        symmetry_policy::none no_symm;
        size_t no_symm_size = no_symm.size(ElementPolicy::range, rank_);
        matrix_t out(boost::extents[no_symm_size][no_symm_size]);
        indices_t i_ind(rank_);
        for (size_t i = 0; i < size(); ++i, advance_ind(i_ind)) {
            do {
                size_t i_out = ElementPolicy::rearranged_index(i_ind);
                indices_t j_ind(rank_);
                for (size_t j = 0; j < size(); ++j, advance_ind(j_ind)) {
                    do {
                        size_t j_out = ElementPolicy::rearranged_index(j_ind);
                        out[i_out][j_out] = c[i][j] / (weights[i] * weights[j]);
                    } while (unsymmetrize && transform_ind(j_ind));
                }
            } while (unsymmetrize && transform_ind(i_ind));
        }
        return out;
    }

    std::pair<matrix_t, matrix_t> block_structure (matrix_t const& c) const {
        size_t block_range = combinatorics::ipow(ElementPolicy::n_block, rank_);
        size_t block_size = combinatorics::ipow(ElementPolicy::range / ElementPolicy::n_block, rank_);
        std::pair<matrix_t, matrix_t> blocks {
            matrix_t (boost::extents[block_range][block_range]),
            matrix_t (boost::extents[block_range][block_range])
        };
        boost::multi_array<block_reduction::norm<2>,2> block_2norms(boost::extents[block_range][block_range]);
        boost::multi_array<block_reduction::sum,2> block_sums(boost::extents[block_range][block_range]);
        indices_t i_ind(rank_);
        for (size_t i = 0; i < size(); ++i, advance_ind(i_ind)) {
            do {
                size_t i_out = ElementPolicy::rearranged_index(i_ind);
                indices_t j_ind(rank_);
                for (size_t j = 0; j < size(); ++j, advance_ind(j_ind)) {
                    do {
                        size_t j_out = ElementPolicy::rearranged_index(j_ind);
                        block_2norms[i_out / block_size][j_out / block_size] += c[i][j];
                        block_sums[i_out / block_size][j_out / block_size] += c[i][j];
                    } while (unsymmetrize && transform_ind(j_ind));
                }
            } while (unsymmetrize && transform_ind(i_ind));
        }
        for (size_t i = 0; i < block_range; ++i) {
            for (size_t j = 0; j < block_range; ++j) {
                blocks.first[i][j] = block_2norms[i][j];
                blocks.second[i][j] = block_sums[i][j];
            }
        }
        return blocks;
    }

    virtual size_t range () const override {
        return ElementPolicy::range;
    }

    virtual size_t rank () const override {
        return rank_;
    }

    virtual indices_t block_indices(indices_t const& ind) const override {
        indices_t cind;
        cind.reserve(ind.size());
        std::transform(ind.begin(), ind.end(), std::back_inserter(cind),
                       [this] (size_t a) { return sublattice(a) * ElementPolicy::n_color + color(a); });
        return cind;
    }

    virtual indices_t component_indices(indices_t const& ind) const override {
        indices_t cind;
        cind.reserve(ind.size());
        std::transform(ind.begin(), ind.end(), std::back_inserter(cind),
                       [this] (size_t a) { return component(a); });
        return cind;
    }

    virtual contraction_matrix_t contraction_matrix(std::vector<detail::contraction> const& cs,
                                                    indices_t const& i_col_ind,
                                                    indices_t const& j_col_ind) const override {
        size_t ts = size();
        contraction_matrix_t a(ts * ts, cs.size());
        indices_t i_ind(rank_);
        for (size_t i = 0; i < size(); ++i, advance_ind(i_ind)) {
            bool i_color_match = std::equal(i_col_ind.begin(), i_col_ind.end(), block_indices(i_ind).begin());
            indices_t j_ind(rank_);
            for (size_t j = 0; j < size(); ++j, advance_ind(j_ind)) {
                bool j_color_match = std::equal(j_col_ind.begin(), j_col_ind.end(), block_indices(j_ind).begin());
                for (size_t k = 0; k < cs.size(); ++k) {
                    if (i_color_match && j_color_match)
                        a(i * ts + j, k) = cs[k](component_indices(i_ind),
                                                 component_indices(j_ind)) ? 1 : 0;
                    else
                        a(i * ts + j, k) = 0.;
                }
            }
        }
        return a;
    }

    virtual std::set<indices_t> all_block_indices () const override {
        std::set<indices_t> b;
        indices_t i_ind(rank());
        for (size_t i = 0; i < size(); ++i, advance_ind(i_ind)) {
            b.insert(block_indices(i_ind));
        }
        return b;
    }

private:
    using ElementPolicy::color;
    using ElementPolicy::component;
    using ElementPolicy::sublattice;

    void advance_ind (indices_t & ind) const {
        SymmetryPolicy::advance_ind(ind, ElementPolicy::range);
    }
    using SymmetryPolicy::transform_ind;

    size_t rank_;
    bool unsymmetrize;
    std::vector<double> weights;
};

template <typename ElementPolicy>
struct site_resolved_rank1_config_policy : public config_policy, private ElementPolicy {
    site_resolved_rank1_config_policy (size_t N) : n_sites(N) {}

    virtual size_t size () const override {
        return ElementPolicy::range * n_sites;
    }

    virtual std::vector<double> configuration (config_array const& R) const override {
        std::vector<double> v;
        v.reserve(size());
        for (local_state const& site : R) {
            for (size_t a = 0; a < ElementPolicy::range; ++a) {
                v.push_back(site(color(a), component(a)));
            }
        }
        return v;
    }

    virtual matrix_t rearrange_by_component (matrix_t const& c) const override {
        matrix_t out(boost::extents[size()][size()]);
        for (size_t i = 0; i < size(); ++i) {
            size_t i_out = (i % ElementPolicy::range) * n_sites + (i / ElementPolicy::range);
            for (size_t j = 0; j < size(); ++j) {
                size_t j_out = (j % ElementPolicy::range) * n_sites + (j / ElementPolicy::range);
                out[i_out][j_out] = c[i][j];
            }
        }
        return out;
    }

    std::pair<matrix_t, matrix_t> block_structure (matrix_t const& c) const {
        size_t block_range = ElementPolicy::n_color * n_sites;
        size_t block_size = ElementPolicy::range / ElementPolicy::n_color;
        std::pair<matrix_t, matrix_t> blocks {
            matrix_t (boost::extents[block_range][block_range]),
                matrix_t (boost::extents[block_range][block_range])
                };
        boost::multi_array<block_reduction::norm<2>,2> block_2norms(boost::extents[block_range][block_range]);
        boost::multi_array<block_reduction::sum,2> block_sums(boost::extents[block_range][block_range]);
        for (size_t i = 0; i < size(); ++i) {
            size_t i_out = (i % ElementPolicy::range) * n_sites + (i / ElementPolicy::range);
            for (size_t j = 0; j < size(); ++j) {
                size_t j_out = (j % ElementPolicy::range) * n_sites + (j / ElementPolicy::range);
                block_2norms[i_out / block_size][j_out / block_size] += c[i][j];
                block_sums[i_out / block_size][j_out / block_size] += c[i][j];
            }
        }
        for (size_t i = 0; i < block_range; ++i) {
            for (size_t j = 0; j < block_range; ++j) {
                blocks.first[i][j] = block_2norms[i][j];
                blocks.second[i][j] = block_sums[i][j];
            }
        }
        return blocks;
    }

    virtual size_t range () const override {
        return size();
    }

    virtual size_t rank () const override {
        return 1;
    }

    virtual indices_t block_indices(indices_t const& ind) const override {
        indices_t cind;
        cind.reserve(ind.size());
        std::transform(ind.begin(), ind.end(), std::back_inserter(cind),
                       [this] (size_t a) { return color(a); });
        return cind;
    }

    virtual indices_t component_indices(indices_t const& ind) const override {
        indices_t cind;
        cind.reserve(ind.size());
        std::transform(ind.begin(), ind.end(), std::back_inserter(cind),
                       [this] (size_t a) { return component(a); });
        return cind;
    }

    virtual contraction_matrix_t contraction_matrix(std::vector<detail::contraction> const& cs,
                                                    indices_t const& i_col_ind,
                                                    indices_t const& j_col_ind) const override {
        size_t ts = size();
        contraction_matrix_t a(ts * ts, cs.size());
        // TODO
        // indices_t i_ind(rank_);
        // for (size_t i = 0; i < size(); ++i, advance_ind(i_ind)) {
        //     indices_t j_ind(rank_);
        //     for (size_t j = 0; j < size(); ++j, advance_ind(j_ind)) {
        //         for (size_t k = 0; k < cs.size(); ++k) {
        //             a(i * ts + j, k) = cs[k](i_ind, j_ind);
        //         }
        //     }
        // }
        return a;
    }

    virtual std::set<indices_t> all_block_indices () const override {
        std::set<indices_t> b;
        // TODO
        // indices_t i_ind(rank());
        // for (size_t i = 0; i < size(); ++i, advance_ind(i_ind)) {
        //     b.insert(block_indices(i_ind));
        // }
        return b;
    }

private:
    using ElementPolicy::color;
    using ElementPolicy::component;

    size_t n_sites;
};
