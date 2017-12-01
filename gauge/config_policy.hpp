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
    };

}


namespace symmetry_policy {

    struct none {
        size_t size (size_t range, size_t rank) const {
            return combinatorics::ipow(range, rank);
        }

        void advance_ind (std::vector<size_t> & ind, size_t range) const {
            auto it = ind.begin();
            ++(*it);
            while (*it == range) {
                ++it;
                if (it == ind.end())
                    break;
                ++(*it);
                std::reverse_iterator<decltype(it)> rit(it);
                while (rit != ind.rend()) {
                    *rit = 0;
                    ++rit;
                }
            }
        }
    };

    struct symmetrized {
        size_t size (size_t range, size_t rank) const {
            return combinatorics::binomial(rank + range - 1, rank);
        }

        void advance_ind (std::vector<size_t> & ind, size_t range) const {
            auto it = ind.begin();
            ++(*it);
            while (*it == range) {
                ++it;
                if (it == ind.end())
                    break;
                ++(*it);
                std::reverse_iterator<decltype(it)> rit(it);
                while (rit != ind.rend()) {
                    *rit = *it;
                    ++rit;
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

private:
    using ElementPolicy::color;
    using ElementPolicy::component;

    void advance_ind (std::vector<size_t> & ind) const {
        SymmetryPolicy::advance_ind(ind, ElementPolicy::range);
    }

    size_t rank;
};
