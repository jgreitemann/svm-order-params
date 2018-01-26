#pragma once

#include "config_policy.hpp"

#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include <boost/multi_array.hpp>


struct delta_rule {
    bool operator() (std::vector<size_t> const& i_ind, std::vector<size_t> const& j_ind) const;
    size_t rank () const;
    delta_rule (std::string const& lhs, std::string const& rhs);
private:
    std::pair<std::string, std::string> pattern;
};

struct contraction {
    double operator() (std::vector<size_t> const& i_ind, std::vector<size_t> const& j_ind) const;
    contraction (double, delta_rule &&);
    delta_rule const& rule () const;
private:
    double weight;
    delta_rule rule_;
};

struct tensor_factory {
    template <typename ElementPolicy>
    boost::multi_array<double, 2> get () const {
        return operator() (ElementPolicy::range);
    }
    boost::multi_array<double, 2> get (std::unique_ptr<config_policy> const& cpol) const;
    tensor_factory (std::vector<contraction> && bc, std::vector<contraction> && cc);
private:
    std::vector<contraction> block_contractions;
    std::vector<contraction> component_contractions;
};

const std::map<std::string, tensor_factory> exact_tensor = {
    {
        "Dinfh",
        {
            {
                {   1./9,  {"aa", "bb"}},
                {  -1./3,  {"aa", "22"}},
                {  -1./3,  {"22", "aa"}},
                {     1.,  {"22", "22"}}
            },
            {
                {     1.,  {"ab", "ab"}},
                {     1.,  {"ab", "ba"}},
                {  -2./3,  {"aa", "bb"}}
            }
        }
    }
};