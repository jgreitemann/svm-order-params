#pragma once

#include "config_policy.hpp"

#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include <boost/multi_array.hpp>


struct index_rule {
    using rule_ptr = std::unique_ptr<index_rule>;
    virtual bool operator() (indices_t const&, indices_t const&) const = 0;
    virtual rule_ptr clone () const = 0;
};

using rule_ptr = index_rule::rule_ptr;

struct delta_rule : public index_rule {
    virtual bool operator() (indices_t const& i_ind, indices_t const& j_ind) const override;
    virtual rule_ptr clone () const override;
    delta_rule (std::string const& lhs, std::string const& rhs);
private:
    std::pair<std::string, std::string> pattern;
};

rule_ptr make_delta (std::string const& lhs, std::string const& rhs);

struct distinct_rule : public index_rule {
    virtual bool operator() (indices_t const& i_ind, indices_t const& j_ind) const override;
    virtual rule_ptr clone () const override;
};

rule_ptr make_distinct ();

struct contraction {
    double operator() (indices_t const& i_ind, indices_t const& j_ind) const;
    contraction (double, rule_ptr &&);
    contraction (contraction const&);
private:
    double weight;
    rule_ptr rule_;
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
                {   1./9, make_delta("aa", "bb")},
                {  -1./3, make_delta("aa", "22")},
                {  -1./3, make_delta("22", "aa")},
                {     1., make_delta("22", "22")}
            },
            {
                {     1., make_delta("ab", "ab")},
                {     1., make_delta("ab", "ba")},
                {  -2./3, make_delta("aa", "bb")}
            }
        }
    },
    {
        "Td",
        {
            {
                {     1.,  make_distinct()}
            },
            {
                {     1.,  make_delta("abc", "abc")},
                {     1.,  make_delta("abc", "bca")},
                {     1.,  make_delta("abc", "cab")},
                {     1.,  make_delta("abc", "bac")},
                {     1.,  make_delta("abc", "acb")},
                {     1.,  make_delta("abc", "cba")},
                {  -2./5,  make_delta("aac", "bbc")},
                {  -2./5,  make_delta("aac", "bcb")},
                {  -2./5,  make_delta("aac", "cbb")},
                {  -2./5,  make_delta("aca", "bbc")},
                {  -2./5,  make_delta("aca", "bcb")},
                {  -2./5,  make_delta("aca", "cbb")},
                {  -2./5,  make_delta("caa", "bbc")},
                {  -2./5,  make_delta("caa", "bcb")},
                {  -2./5,  make_delta("caa", "cbb")}
            }
        }
    }
};
