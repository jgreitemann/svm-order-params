#pragma once

#include "combinatorics.hpp"

#include <iterator>
#include <vector>

#include <boost/multi_array.hpp>

#include <Eigen/Dense>


namespace element_policy {

    struct uniaxial {
        static const size_t range = 3;
        inline size_t color(size_t index) const { return 2; }
        inline size_t component(size_t index) const { return index; }
    };

    struct triad {
        static const size_t range = 9;
        inline size_t color(size_t index) const { return index / 3; }
        inline size_t component(size_t index) const { return index % 3; }

        size_t rearranged_index (std::vector<size_t> const& ind) const {
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

}


namespace symmetry_policy {

    struct none {
        size_t size (size_t range, size_t rank) const {
            return combinatorics::ipow(range, rank);
        }

        void advance_ind (std::vector<size_t> & ind, size_t range) const {
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
    };

    struct symmetrized {
        size_t size (size_t range, size_t rank) const {
            return combinatorics::binomial(rank + range - 1, rank);
        }

        void advance_ind (std::vector<size_t> & ind, size_t range) const {
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
    };

}


struct config_policy {
    typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> local_state;
    typedef boost::multi_array<local_state, 1> config_array;
    typedef boost::multi_array<double, 2> matrix_t;

    virtual size_t size () const = 0;
    virtual std::vector<double> configuration (config_array const&) const = 0;

    virtual matrix_t rearrange_by_component (matrix_t const& c) const = 0;
};

template <typename ElementPolicy, typename SymmetryPolicy>
struct gauge_config_policy : public config_policy, private ElementPolicy, SymmetryPolicy {
    gauge_config_policy (size_t rank) : rank(rank) {}

    virtual size_t size () const override {
        return SymmetryPolicy::size(ElementPolicy::range, rank);
    }

    virtual std::vector<double> configuration (config_array const& R) const override {
        std::vector<double> v(size());
        std::vector<size_t> ind(rank);
        for (double & elem : v) {
            for (local_state const& site : R) {
                double prod = 1;
                for (size_t a : ind)
                    prod *= site(color(a), component(a));
                elem += prod;
            }
            elem /= R.size();

            advance_ind(ind);
        }
        return v;
    }

    virtual matrix_t rearrange_by_component (matrix_t const& c) const override {
        symmetry_policy::none no_symm;
        size_t no_symm_size = no_symm.size(ElementPolicy::range, rank);
        if constexpr(std::is_same<ElementPolicy, element_policy::triad>::value) {
            matrix_t out(boost::extents[no_symm_size][no_symm_size]);
            std::vector<size_t> i_ind(rank);
            for (size_t i = 0; i < size(); ++i, advance_ind(i_ind)) {
                size_t i_out = ElementPolicy::rearranged_index(i_ind);
                std::vector<size_t> j_ind(rank);
                for (size_t j = 0; j < size(); ++j, advance_ind(j_ind)) {
                    size_t j_out = ElementPolicy::rearranged_index(j_ind);
                    out[i_out][j_out] = c[i][j];
                }
            }
            return out;
        } else {
            return c;
        }
    }

private:
    using ElementPolicy::color;
    using ElementPolicy::component;

    void advance_ind (std::vector<size_t> & ind) const {
        SymmetryPolicy::advance_ind(ind, ElementPolicy::range);
    }

    size_t rank;
};
