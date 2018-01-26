#include "results.hpp"
#include "config_policy.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>


delta_rule::delta_rule (std::string const& lhs, std::string const& rhs)
    : pattern {lhs, rhs} {}

bool delta_rule::operator() (std::vector<size_t> const& i_ind, std::vector<size_t> const& j_ind) const {
    if (i_ind.size() != pattern.first.size() || j_ind.size() != pattern.second.size())
        throw std::runtime_error("index ranks don't match delta rule pattern");

    std::map<char, size_t> assignments;
    auto idx_it = i_ind.begin();
    auto c_it = pattern.first.begin();
    while (idx_it != j_ind.end()) {
        auto find = assignments.find(*c_it);
        if (find == assignments.end()) {
            assignments[*c_it] = *idx_it;
        } else {
            if (find->second != *idx_it)
                return false;
        }

        ++idx_it;
        ++c_it;
        if (idx_it == i_ind.end()) {
            idx_it = j_ind.begin();
            c_it = pattern.second.begin();
        }
    }
    return true;
}

size_t delta_rule::rank () const {
    if (pattern.first.size() != pattern.second.size())
        throw std::runtime_error("delta rule pattern has different ranks");
    return pattern.first.size();
}

tensor_factory::tensor_factory (std::initializer_list<std::pair<double, delta_rule>> il) {
    std::copy(il.begin(), il.end(), std::back_inserter(contractions));
}

boost::multi_array<double, 2> tensor_factory::get (size_t range) const {
    size_t rank = contractions.begin()->second.rank();
    for (auto const& [p, c] : contractions)
        if (c.rank() != rank)
            throw std::runtime_error("inconsistent ranks across different"
                                     "contractions in the same tensor factory");
    symmetry_policy::none symm;
    size_t size = symm.size(range, rank);
    boost::multi_array<double, 2> res(boost::extents[size][size]);
    std::vector<size_t> i_ind(rank);
    for (auto row : res) {
        std::vector<size_t> j_ind(rank);
        for (auto & elem : row) {
            elem = 0;
            for (auto const& [p, c] : contractions)
                if (c(i_ind, j_ind))
                    elem += p;
            symm.advance_ind(j_ind, range);
        }
        symm.advance_ind(i_ind, range);
    }
    return res;
}
