#pragma once

#include "svm-wrapper.hpp"
#include "hdf5_serialization.hpp"

#include <tuple>

#include <alps/mc/mcbase.hpp>


void define_test_parameters(alps::params & parameters) {
    if (!parameters.is_restored()) {
        parameters
            .define<double>("test.a.J1", 0., "minimum temperature in test")
            .define<double>("test.a.J3", 10., "maximum temperature in test")
            .define<double>("test.b.J1", 0., "minimum temperature in test")
            .define<double>("test.b.J3", 10., "maximum temperature in test")
            .define<size_t>("test.N_scan", 10, "number of temperatures to test at")
            .define<std::string>("test.filename", "", "test output file name")
            .define<std::string>("test.txtname", "", "test output txt name")
            ;
    }
}

template <class Simulation>
class test_adapter : public Simulation {
public:
    typedef alps::mcbase::parameters_type parameters_type;

    using kernel_t = svm::kernel::polynomial<2>;
    using phase_label = typename Simulation::phase_label;
    using model_t = svm::model<kernel_t, phase_label>;
    using problem_t = svm::problem<kernel_t>;

    test_adapter (parameters_type & parms, std::size_t seed_offset = 0)
        : Simulation(parms, seed_offset)
    {
        std::string arname = parms.get_archive_name();

#pragma omp critical
        {
            alps::hdf5::archive ar(arname, "r");

            svm::model_serializer<svm::hdf5_tag, model_t> serial(model);
            ar["model"] >> serial;
        }

        measurements << alps::accumulators::FullBinningAccumulator<std::vector<double>>("SVM")
                     << alps::accumulators::FullBinningAccumulator<double>("label");
    }

    virtual void measure () override {
        Simulation::measure();
        if (Simulation::is_thermalized()) {
            auto res = model(svm::dataset(Simulation::configuration()));
            measurements["label"] << double(res.first);

            std::vector<double> decs(model_t::nr_classifiers);
            std::copy((double*)&(res.second),
                      (double*)&(res.second) + model_t::nr_classifiers,
                      decs.begin());
            measurements["SVM"] << decs;
        }
    }

private:
    using Simulation::measurements;
    model_t model;
};
