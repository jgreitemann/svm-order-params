#include "gauge.hpp"
#include "svm-wrapper.hpp"
#include "hdf5_serialization.hpp"

#include <array>
#include <iostream>

#include <alps/hdf5.hpp>
#include <alps/utilities/fs/remove_extensions.hpp>

#include <boost/multi_array.hpp>


using sim_type = gauge_sim;
using kernel_t = svm::kernel::polynomial<2>;

int main(int argc, char** argv) {
    try {
        alps::params parameters(argc, argv);
        sim_type::define_parameters(parameters);

        if (parameters.help_requested(std::cout) ||
            parameters.has_missing(std::cout)) {
            return 1;
        }

        std::string arname = parameters.get_archive_name();

        svm::model<kernel_t> model;
        {
            alps::hdf5::archive ar(arname, "r");
            svm::model_serializer<svm::hdf5_tag, svm::model<kernel_t>> serial(model);
            ar["model"] >> serial;
        }

        svm::tensor_introspector<kernel_t, 2> coeff(model);

        std::string outname = alps::fs::remove_extensions(arname) + ".coeffs.txt";
        std::ofstream os(outname);
        boost::multi_array<double,2> coeffs(boost::extents[model.dim()][model.dim()]);
#pragma omp parallel for
        for (size_t i = 0; i < model.dim(); ++i) {
            for (size_t j = 0; j < model.dim(); ++j) {
                coeffs[i][j] = coeff.tensor({i, j});
            }
        }
        for (size_t i = 0; i < model.dim(); ++i) {
            for (size_t j = 0; j < model.dim(); ++j) {
                os << coeffs[i][j] << '\t';
            }
            os << '\n';
        }

        return 0;
    } catch (const std::runtime_error& exc) {
        std::cout << "Exception caught: " << exc.what() << std::endl;
        return 2;
    } catch (...) {
        std::cout << "Unknown exception caught." << std::endl;
        return 2;
    }
}
