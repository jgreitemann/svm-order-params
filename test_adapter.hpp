// SVM Order Parameters for Hidden Spin Order
// Copyright (C) 2018-2019  Jonas Greitemann, Ke Liu, and Lode Pollet

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

#include "svm-wrapper.hpp"
#include "hdf5_serialization.hpp"

#include <tuple>

#include <alps/mc/mcbase.hpp>


template <class Simulation>
class test_adapter : public Simulation {
public:
    typedef alps::mcbase::parameters_type parameters_type;

    using kernel_t = svm::kernel::polynomial<2>;
    using phase_label = typename Simulation::phase_label;
    using model_t = svm::model<kernel_t, phase_label>;
    using problem_t = svm::problem<kernel_t>;
    using introspec_t = svm::tensor_introspector<typename model_t::classifier_type, 2>;

    using config_policy_t = typename Simulation::template config_policy_type<introspec_t>;

    static void define_test_parameters(alps::params & parameters) {
        if (!parameters.is_restored()) {
            parameters
                .define<std::string>("test.filename", "", "test output file name")
                .define<std::string>("test.txtname", "", "test output txt name")
                ;
            Simulation::test_sweep_type::define_parameters(parameters, "test.");
        }
    }

    static void define_parameters(alps::params & parameters) {
        Simulation::define_parameters(parameters);
        define_test_parameters(parameters);
    }

    test_adapter(parameters_type & parms, std::size_t seed_offset = 0)
        : Simulation(parms, seed_offset)
        , confpol(Simulation::template config_policy_from_parameters<introspec_t>(parms))
    {
        std::string arname = parms["outputfile"];

        {
            alps::hdf5::archive ar(arname, "r");

            svm::model_serializer<svm::hdf5_tag, model_t> serial(model);
            ar["model"] >> serial;
        }

        measurements()
        << alps::accumulators::FullBinningAccumulator<std::vector<double>>("SVM")
        << alps::accumulators::FullBinningAccumulator<std::vector<double>>("SVM^2")
        << alps::accumulators::FullBinningAccumulator<double>("label");
    }

    virtual void measure () override {
        Simulation::measure();
        if (Simulation::is_thermalized()) {
            auto res = model(svm::dataset(confpol->configuration(Simulation::configuration())));
            measurements()["label"] << double(res.first);

            // measure decision functions
            auto decs = svm::detail::container_factory<std::vector<double>>::copy(res.second);
            measurements()["SVM"] << decs;

            // measure decision function squares
            std::transform(decs.begin(), decs.end(), decs.begin(), decs.begin(),
                std::multiplies<>{});
            measurements()["SVM^2"] << decs;
        }
    }

    virtual void reset_sweeps(bool skip_therm = false) override {
        Simulation::reset_sweeps(skip_therm);
    }

protected:
    using Simulation::measurements;
private:
    model_t model;
    std::unique_ptr<config_policy_t> confpol;
};
