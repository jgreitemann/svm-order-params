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

#include "config_serialization.hpp"
#include "hdf5_serialization.hpp"
#include "training_adapter.hpp"

#include <deque>
#include <sstream>

#include <boost/multi_array.hpp>

template <class Simulation>
class procrastination_adapter : public training_adapter<Simulation> {
public:
    using Base = training_adapter<Simulation>;
    using typename Base::parameters_type;

    using typename Base::phase_point;
    using typename Base::phase_classifier;
    using typename Base::label_t;
    using typename Base::phase_sweep_policy_type;

    using typename Base::kernel_t;
    using typename Base::problem_t;
    using typename Base::model_t;
    using typename Base::introspec_t;

    using typename Base::config_policy_t;

    using config_array = typename config_policy_t::config_array;

    procrastination_adapter(parameters_type & parms,
                            double const& global_progress,
                            std::size_t seed_offset = 0)
        : Base(parms, global_progress, seed_offset)
    {
    }

    using alps::mcbase::save;
    virtual void save (alps::hdf5::archive & ar) const override {
        Base::save(ar);
        if (!config_buffer.empty()) {
            config_serializer<config_array> serializer;
            size_t config_size = [&] {
                auto config = Simulation::configuration();
                std::vector<double> dummy;
                serializer.serialize(config, std::back_inserter(dummy));
                return dummy.size();
            }();
            boost::multi_array<double, 2> buffer_multi_array(
                boost::extents[config_buffer.size()][config_size
                    + phase_point::label_dim]);
            auto row_it = buffer_multi_array.begin();
            for (auto const& conf : config_buffer) {
                auto col_it = std::copy(conf.second.begin(), conf.second.end(),
                    (row_it++)->begin());
                serializer.serialize(conf.first, col_it);
            }
            ar["training/config_buffer"] << buffer_multi_array;
        }
    }

    using alps::mcbase::load;
    virtual void load (alps::hdf5::archive & ar) override {
        using namespace std::literals::string_literals;
        Base::load(ar);

        boost::multi_array<double, 2> buffer_multi_array;
        ar["training/config_buffer"] >> buffer_multi_array;
        config_serializer<config_array> serializer;
        for (auto const& row : buffer_multi_array) {
            config_buffer.push_back({Simulation::configuration(),
                {row.begin()}});
            auto col_it = row.begin() + phase_point::label_dim;
            serializer.deserialize(col_it, config_buffer.back().first);
        }
    }

    problem_t surrender_problem () {
        problem_t problem(confpol->size());
        while (!config_buffer.empty()) {
            auto & conf = config_buffer.front().first;
            auto & point = config_buffer.front().second;
            problem.add_sample(confpol->configuration(conf), point);
            config_buffer.pop_front();
        }
        return problem;
    }

protected:
    virtual void sample_config() override {
        config_buffer.emplace_back(Simulation::configuration(),
                                   Simulation::phase_space_point());
    }

private:
    std::deque<std::pair<config_array, phase_point>> config_buffer;

    using Base::confpol;
};
