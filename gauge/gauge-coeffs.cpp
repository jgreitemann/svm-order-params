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

        boost::multi_array<double,2> coeffs(boost::extents[model.dim()][model.dim()]);
#pragma omp parallel for
        for (size_t i = 0; i < model.dim(); ++i) {
            for (size_t j = 0; j < model.dim(); ++j) {
                coeffs[i][j] = coeff.tensor({i, j});
            }
        }

        // rearrange elements in blocks by components
        std::unique_ptr<config_policy> confpol =
            sim_type::config_policy_from_parameters(parameters);
        {
            auto rearranged_coeffs = confpol->rearrange_by_component(coeffs);
            std::string outname = alps::fs::remove_extensions(arname) + ".coeffs.txt";
            std::ofstream os(outname);
            for (size_t i = 0; i < rearranged_coeffs.shape()[0]; ++i) {
                for (size_t j = 0; j < rearranged_coeffs.shape()[1]; ++j) {
                    os << rearranged_coeffs[i][j] << '\t';
                }
                os << '\n';
            }
        }
        {
            auto block_structure = confpol->block_structure(coeffs);
            std::string outname = alps::fs::remove_extensions(arname) + ".blocks.txt";
            std::ofstream os(outname);
            for (size_t i = 0; i < block_structure.shape()[0]; ++i) {
                for (size_t j = 0; j < block_structure.shape()[1]; ++j) {
                    os << block_structure[i][j] << '\t';
                }
                os << '\n';
            }
        }

        return 0;
    } catch (const std::exception& exc) {
        std::cout << "Exception caught: " << exc.what() << std::endl;
        return 2;
    } catch (...) {
        std::cout << "Unknown exception caught." << std::endl;
        return 2;
    }
}
