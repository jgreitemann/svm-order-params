// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2019  Jonas Greitemann, Ke Liu, and Lode Pollet

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

#include "mpi.hpp"

#include <stdexcept>

namespace pt {
    enum pt_tags {
        query_tag = 287436,
        response_tag,
        partner_tag,
        chosen_partner_tag,
        point_tag,
        weight_tag,
        acceptance_tag
    };

    enum struct status {
        unregistered,
        available,
        primary,
        secondary
    };

    enum struct query_type {
    	status,
    	init,
    	deregister
    };

    template <typename Point,
              typename CoinFlipFunc,
              typename LogWeightFunc,
              typename AttemptUpdateFunc>
    static bool negotiate_update(mpi::communicator const& comm,
        bool initiate_update_if_possible,
        Point this_point,
        CoinFlipFunc && coin_flip,
        LogWeightFunc && log_weight,
        AttemptUpdateFunc && attempt_update)
    {
        mpi::send(comm, static_cast<int>(query_type::status), 0, query_tag);

        int response;
        mpi::receive(comm, response, 0, response_tag);

        switch (static_cast<status>(response)) {
        case status::available:
            if (initiate_update_if_possible) {
                mpi::send(comm, static_cast<int>(query_type::init), 0,
                	query_tag);

                int partner_rank;
                mpi::receive(comm, partner_rank, 0, partner_tag);
                if (partner_rank == -1) {
                	// both neighbors are available, decide for one
                	partner_rank = coin_flip()
	                	? (comm.rank() - 1) : (comm.rank() + 1);
	                mpi::send(comm, partner_rank, 0, chosen_partner_tag);
                } else if (partner_rank == -2) {
                	return false;
                }

                Point other_point;
                mpi::send(comm, this_point.begin(), this_point.end(),
                    partner_rank, point_tag);
                mpi::receive(comm, other_point.begin(), other_point.end(),
                    partner_rank, point_tag);

                double other_weight;
                mpi::receive(comm, other_weight, partner_rank, weight_tag);

                bool acc = attempt_update(other_point, other_weight);
                mpi::send(comm, static_cast<int>(acc), partner_rank,
                	acceptance_tag);
                return acc;
            }
            break;
        case status::secondary:
            {
                int partner_rank;
                mpi::receive(comm, partner_rank, 0, partner_tag);

                Point other_point;
                mpi::receive(comm, other_point.begin(), other_point.end(),
                    partner_rank, point_tag);
                mpi::send(comm, this_point.begin(), this_point.end(),
                    partner_rank, point_tag);

                double this_weight = log_weight(other_point);
                mpi::send(comm, this_weight, partner_rank, weight_tag);

                int acc;
                mpi::receive(comm, acc, partner_rank, acceptance_tag);
                if (acc) {
                    acc = attempt_update(other_point, 1. - this_weight);
                    if (!acc)
                        throw std::runtime_error(
                            "PT update should have been accepted.");
                    return acc;
                }
            }
            break;
        default:
            throw std::runtime_error("illegal response in this context");
        }
        return false;
    }
}

