#pragma once

#include <initializer_list>
#include <map>
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

struct tensor_factory {
    template <typename ElementPolicy>
    boost::multi_array<double, 2> get () const {
        return operator() (ElementPolicy::range);
    }
    boost::multi_array<double, 2> get (size_t range) const;
    tensor_factory (std::initializer_list<std::pair<double, delta_rule>> il);
private:
    std::vector<std::pair<double, delta_rule>> contractions;
};

const std::map<std::string, tensor_factory> exact_tensor = {
    {
        "Dinfh",
        {
            {     1.,  {"ab", "ab"}},
            {     1.,  {"ab", "ba"}},
            {  -2./3,  {"aa", "bb"}}
        }
    }
};
