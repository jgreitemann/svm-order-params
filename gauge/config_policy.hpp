#pragma once

#include "combinatorics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <boost/multi_array.hpp>

#include <Eigen/Dense>


using indices_t = std::vector<size_t>;

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
}


struct config_policy {
    typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> local_state;
    typedef boost::multi_array<local_state, 1> config_array;
    typedef boost::multi_array<double, 2> matrix_t;

    virtual size_t size () const = 0;
    virtual size_t range () const = 0;
    virtual size_t rank () const = 0;
    virtual std::vector<double> configuration (config_array const&) const = 0;

    virtual matrix_t rearrange_by_component (matrix_t const& c) const = 0;
    virtual matrix_t block_structure (matrix_t const& c) const = 0;

    virtual indices_t block_indices(indices_t const& ind) const = 0;
    virtual indices_t component_indices(indices_t const& ind) const = 0;
};

template <typename LatticePolicy, typename SymmetryPolicy,
          typename BlockReduction = block_reduction::norm<1>,
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

    virtual matrix_t block_structure (matrix_t const& c) const override {
        size_t block_range = combinatorics::ipow(ElementPolicy::n_block, rank_);
        size_t block_size = combinatorics::ipow(ElementPolicy::range / ElementPolicy::n_block, rank_);
        matrix_t blocks(boost::extents[block_range][block_range]);
        boost::multi_array<BlockReduction,2> block_norms(boost::extents[block_range][block_range]);
        indices_t i_ind(rank_);
        for (size_t i = 0; i < size(); ++i, advance_ind(i_ind)) {
            do {
                size_t i_out = ElementPolicy::rearranged_index(i_ind);
                indices_t j_ind(rank_);
                for (size_t j = 0; j < size(); ++j, advance_ind(j_ind)) {
                    do {
                        size_t j_out = ElementPolicy::rearranged_index(j_ind);
                        block_norms[i_out / block_size][j_out / block_size]
                            += c[i][j] / (weights[i] * weights[j]);
                    } while (unsymmetrize && transform_ind(j_ind));
                }
            } while (unsymmetrize && transform_ind(i_ind));
        }
        for (size_t i = 0; i < block_range; ++i)
            for (size_t j = 0; j < block_range; ++j)
                blocks[i][j] = block_norms[i][j];
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

template <typename ElementPolicy, typename BlockReduction = block_reduction::norm<1>>
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

    virtual matrix_t block_structure (matrix_t const& c) const override {
        size_t block_range = ElementPolicy::n_color * n_sites;
        size_t block_size = ElementPolicy::range / ElementPolicy::n_color;
        matrix_t blocks(boost::extents[block_range][block_range]);
        boost::multi_array<BlockReduction,2> block_norms(boost::extents[block_range][block_range]);
        for (size_t i = 0; i < size(); ++i) {
            size_t i_out = (i % ElementPolicy::range) * n_sites + (i / ElementPolicy::range);
            for (size_t j = 0; j < size(); ++j) {
                size_t j_out = (j % ElementPolicy::range) * n_sites + (j / ElementPolicy::range);
                block_norms[i_out / block_size][j_out / block_size] += c[i][j];
            }
        }
        for (size_t i = 0; i < block_range; ++i)
            for (size_t j = 0; j < block_range; ++j)
                blocks[i][j] = block_norms[i][j];
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

private:
    using ElementPolicy::color;
    using ElementPolicy::component;

    size_t n_sites;
};
