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
#include "label.hpp"

#include <random>

#include <alps/params.hpp>


namespace phase_space {

    namespace label {

        SVM_LABEL_BEGIN(binary, 2)
        SVM_LABEL_ADD(ORDERED)
        SVM_LABEL_ADD(DISORDERED)
        SVM_LABEL_END()
        
    };

    namespace point {

        struct temperature {
            static const size_t label_dim = 1;

            temperature(double temp) : temp(temp) {}

            template <class Iterator>
            temperature(Iterator begin) : temp(*begin) {}

            double const * begin() const { return &temp; }
            double const * end() const { return &temp + 1; }
            
            double temp;
        };

    }

    namespace classifier {
        
        struct critical_temperature {
            using point_type = point::temperature;
            using label_type = label::binary::label;

            critical_temperature(alps::params const& params);
            label_type operator() (point_type pp);
        private:
            double temp_crit;
        };

    }

    namespace sweep {

        void define_parameters (alps::params & params);

        template <typename Point, typename RNG = std::mt19937>
        struct policy {
            using point_type = Point;
            using rng_type = RNG;

            virtual bool yield (point_type & point, rng_type & rng) = 0;
        };

        struct gaussian_temperatures : public policy<point::temperature> {
            gaussian_temperatures (alps::params const& params);
            virtual bool yield (point_type & point, rng_type & rng) final override;
        private:
            double temp_center;
            double temp_min;
            double temp_max;
            double temp_step;
            double temp_sigma_sq;
        };

        struct uniform_temperatures : public policy<point::temperature> {
            uniform_temperatures (alps::params const& params);
            virtual bool yield (point_type & point, rng_type & rng) final override;
        private:
            double temp_min;
            double temp_max;
            double temp_step;
        };
        
    }

}
