// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2018  Jonas Greitemann, Ke Liu, and Lode Pollet

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include "indices.hpp"

#include <iostream>
#include <utility>
#include <vector>


namespace detail {
    struct contraction {
        typedef std::vector<size_t> ep_type;

        contraction (ep_type &&e) : endpoints(std::forward<ep_type>(e)) {}
        bool is_self_contraction () const;
        bool operator() (indices_t const&, indices_t const&) const;

    private:
        ep_type endpoints;
        friend std::ostream & operator<< (std::ostream &, contraction const&);
    };

    std::ostream & operator<< (std::ostream &, contraction const&);
    std::vector<contraction> get_contractions(size_t rank);

}
