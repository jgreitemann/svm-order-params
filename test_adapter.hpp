#pragma once

#include "svm-wrapper.hpp"
#include "hdf5_serialization.hpp"

#include <tuple>

#include <alps/mc/mcbase.hpp>
#include <alps/utilities/fs/get_basename.hpp>
#include <alps/utilities/fs/remove_extensions.hpp>


void define_test_parameters(alps::params & parameters) {
    if (!parameters.is_restored()) {
        parameters
            .define<double>("test.temp_min", 0., "minimum temperature in test")
            .define<double>("test.temp_max", 10., "maximum temperature in test")
            .define<size_t>("test.N_temp", 10, "number of temperatures to test at")
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
    using problem_t = svm::problem<kernel_t>;

    test_adapter (parameters_type & parms, std::size_t seed_offset = 0)
        : Simulation(parms, seed_offset)
    {
        std::string arname = parms.get_archive_name();

        alps::hdf5::archive ar(arname, "r");

        svm::model_serializer<svm::hdf5_tag, svm::model<kernel_t>> serial(model);
        ar["model"] >> serial;

        measurements << alps::accumulators::FullBinningAccumulator<double>("SVM")
                     << alps::accumulators::FullBinningAccumulator<double>("ordered");
    }

    virtual void measure () override {
        Simulation::measure();
        if (Simulation::is_thermalized()) {
            double phase, dec;
            std::tie(phase, dec) = model(svm::dataset(Simulation::configuration()));
            measurements["SVM"] << dec;
            measurements["ordered"] << (phase > 0);
        }
    }

private:
    using Simulation::measurements;
    svm::model<kernel_t> model;
};
