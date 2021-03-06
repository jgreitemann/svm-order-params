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

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <alps/mc/mcbase.hpp>

#include <svm/svm.hpp>
#include <svm/serialization/hdf5.hpp>

#include <tksvm/phase_space/sweep.hpp>


namespace tksvm {

template <class Simulation>
class test_adapter : public Simulation {
public:
    typedef alps::mcbase::parameters_type parameters_type;

    using kernel_t = svm::kernel::polynomial<2>;
    using phase_point = typename Simulation::phase_point;
    using phase_label = typename phase_space::classifier::policy<phase_point>::label_type;
    using model_t = svm::model<kernel_t, phase_label>;
    using problem_t = svm::problem<kernel_t>;
    using introspec_t = svm::tensor_introspector<typename model_t::classifier_type, 2>;

    using config_policy_t = typename Simulation::template config_policy_type<introspec_t>;

    static void define_test_parameters(alps::params & parameters) {
        if (!parameters.is_restored()) {
            phase_space::sweep::define_parameters<phase_point>(parameters, "test.");
            parameters
                .define<std::string>("test.filename", "", "test output file name")
                .define<std::string>("test.txtname", "", "test output txt name")
                ;
        }
    }

    static void define_parameters(alps::params & parameters) {
        Simulation::define_parameters(parameters);
        define_test_parameters(parameters);
    }

    test_adapter(parameters_type & params, std::size_t seed_offset = 0)
        : Simulation(params, seed_offset)
        , confpol(Simulation::template config_policy_from_parameters<introspec_t>(params))
    {
        if (params.is_restored()
            && alps::origin_name(params) != params["test.filename"])
        {
            load_model(params["outputfile"]);
            measurements()
            << alps::accumulators::FullBinningAccumulator<std::vector<double>>("SVM")
            << alps::accumulators::FullBinningAccumulator<std::vector<double>>("SVM^2")
            << alps::accumulators::FullBinningAccumulator<double>("label");
        }
    }

    virtual void measure () override {
        Simulation::measure();
        if (has_model() && Simulation::is_thermalized() && Simulation::fraction_completed() < 1.) {
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
        measurements().reset();
    }

    virtual void save (alps::hdf5::archive & ar) const override {
        Simulation::save(ar);
        ar["has_model"] << has_model();
    }

    virtual void load (alps::hdf5::archive & ar) override {
        Simulation::load(ar);
        bool has_model;
        ar["has_model"] >> has_model;
        if (has_model)
            load_model(parameters["outputfile"]);
    }

    bool has_model() const {
        return !model.empty();
    }

protected:
    using Simulation::measurements;
    using Simulation::parameters;
private:
    model_t model;
    std::unique_ptr<config_policy_t> confpol;

    void load_model(std::string const& arname) {
        alps::hdf5::archive ar(arname, "r");

        svm::serialization::model_serializer<svm::hdf5_tag, model_t> serial(model);
        ar["model"] >> serial;
    }
};

}
