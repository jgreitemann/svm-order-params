#pragma once

#include "combinatorics.hpp"

#include <iterator>
#include <vector>

#include <boost/multi_array.hpp>

#include <Eigen/Dense>


struct config_policy {
    typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> local_state;
    typedef boost::multi_array<local_state, 1> config_array;

    virtual size_t size () const = 0;
    virtual std::vector<double> configuration (config_array const&) const = 0;
};

template <typename ElementPolicy>
struct full_config_policy : public config_policy, private ElementPolicy {
    full_config_policy (size_t rank) : rank(rank) {}

    virtual size_t size () const override {
        return combinatorics::ipow(ElementPolicy::range, rank);
    }

    virtual std::vector<double> configuration (config_array const& R) const override {
        std::vector<double> v(size());
        std::vector<size_t> ind(rank);
        for (double & elem : v) {
            for (local_state const& site : R) {
                double prod = 1;
                for (size_t a : ind)
                    prod *= get(site, a);
                elem += prod;
            }
            elem /= R.size();

            auto it = ind.begin();
            ++(*it);
            while (*it == ElementPolicy::range) {
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
        return v;
    }

private:
    using ElementPolicy::get;
    size_t rank;
};

template <typename ElementPolicy>
struct symmetrized_config_policy : public config_policy, private ElementPolicy {
    symmetrized_config_policy (size_t rank) : rank(rank) {}

    virtual size_t size () const override {
        return combinatorics::binomial(rank + ElementPolicy::range - 1, rank);
    }

    virtual std::vector<double> configuration (config_array const& R) const override {
        std::vector<double> v(size());
        std::vector<size_t> ind(rank);
        for (double & elem : v) {
            for (local_state const& site : R) {
                double prod = 1;
                for (size_t a : ind)
                    prod *= get(site, a);
                elem += prod;
            }
            elem /= R.size();

            auto it = ind.begin();
            ++(*it);
            while (*it == ElementPolicy::range) {
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
        return v;
    }

private:
    using ElementPolicy::get;
    size_t rank;
};

struct uniaxial_element_policy {
    using local_state = typename config_policy::local_state;

    static const size_t range = 3;
    double get (local_state const& site, size_t index) const {
        return site(2, index);
    }
};

struct triaxial_element_policy {
    using local_state = typename config_policy::local_state;

    static const size_t range = 9;
    double get (local_state const& site, size_t index) const {
        return site(index / 3, index % 3);
    }
};
