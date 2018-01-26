#include "results.hpp"

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
        if ('0' <= *c_it && *c_it <= '9') {
            if (*c_it - '0' != *idx_it)
                return false;
        } else {
            auto find = assignments.find(*c_it);
            if (find == assignments.end()) {
                assignments[*c_it] = *idx_it;
            } else {
                if (find->second != *idx_it)
                    return false;
            }
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

contraction::contraction (double weight, delta_rule && rule)
    : weight(weight), rule_(std::forward<delta_rule>(rule)) {}

double contraction::operator() (std::vector<size_t> const& i_ind, std::vector<size_t> const& j_ind) const {
    if (rule_(i_ind, j_ind))
        return weight;
    return 0.;
}

delta_rule const& contraction::rule () const {
    return rule_;
}

tensor_factory::tensor_factory (std::vector<contraction> && bc,
                                std::vector<contraction> && cc)
    : block_contractions(std::forward<std::vector<contraction>>(bc)),
      component_contractions(std::forward<std::vector<contraction>>(cc)) {}

boost::multi_array<double, 2> tensor_factory::get (std::unique_ptr<config_policy> const& cpol) const {
    size_t rank = cpol->rank();
    for (auto const& c : block_contractions)
        if (c.rule().rank() != rank)
            throw std::runtime_error("inconsistent ranks across block "
                                     "contractions in the same tensor factory");
    for (auto const& c : component_contractions)
        if (c.rule().rank() != rank)
            throw std::runtime_error("inconsistent ranks across component "
                                     "contractions in the same tensor factory");
    symmetry_policy::none symm;
    size_t size = symm.size(cpol->range(), rank);
    boost::multi_array<double, 2> res(boost::extents[size][size]);
    std::vector<size_t> i_ind(rank);
    for (auto row : res) {
        std::vector<size_t> i_ind_block = cpol->block_indices(i_ind);
        std::vector<size_t> i_ind_component = cpol->component_indices(i_ind);
        std::vector<size_t> j_ind(rank);
        for (auto & elem : row) {
            elem = 0;
            std::vector<size_t> j_ind_block = cpol->block_indices(j_ind);
            std::vector<size_t> j_ind_component = cpol->component_indices(j_ind);
            for (auto const& bc : block_contractions)
                for (auto const& cc : component_contractions)
                    elem += (bc(i_ind_block, j_ind_block)
                             * cc(i_ind_component, j_ind_component));
            symm.advance_ind(j_ind, cpol->range());
        }
        symm.advance_ind(i_ind, cpol->range());
    }
    return res;
}
