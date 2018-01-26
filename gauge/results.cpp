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

boost::multi_array<double, 2> tensor_factory::get (size_t range) const {
    size_t rank = block_contractions.begin()->rule().rank();
    for (auto const& c : block_contractions)
        if (c.rule().rank() != rank)
            throw std::runtime_error("inconsistent ranks across different block "
                                     "contractions in the same tensor factory");
    for (auto const& c : component_contractions)
        if (c.rule().rank() != rank)
            throw std::runtime_error("inconsistent ranks across different component "
                                     "contractions in the same tensor factory");
    symmetry_policy::none symm;
    size_t size = symm.size(range, rank);
    boost::multi_array<double, 2> res(boost::extents[size][size]);
    std::vector<size_t> i_ind(rank);
    std::vector<size_t> i_ind_block(rank);
    std::vector<size_t> i_ind_component(rank);
    for (auto row : res) {
        std::vector<size_t> j_ind(rank);
        std::vector<size_t> j_ind_block(rank);
        std::vector<size_t> j_ind_component(rank);
        for (auto & elem : row) {
            elem = 0;
            for (auto const& bc : block_contractions)
                for (auto const& cc : component_contractions)
                    elem += (bc(i_ind_block, j_ind_block)
                             * cc(i_ind_component, j_ind_component));
            symm.advance_ind(j_ind, range);
            std::transform(j_ind.begin(), j_ind.end(), j_ind_block.begin(),
                           [] (size_t a) { return a / 3; });
            std::transform(j_ind.begin(), j_ind.end(), j_ind_component.begin(),
                           [] (size_t a) { return a % 3; });
        }
        symm.advance_ind(i_ind, range);
        std::transform(i_ind.begin(), i_ind.end(), i_ind_block.begin(),
                       [] (size_t a) { return a / 3; });
        std::transform(i_ind.begin(), i_ind.end(), i_ind_component.begin(),
                       [] (size_t a) { return a % 3; });
    }
    return res;
}
