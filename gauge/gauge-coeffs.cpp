#include "gauge.hpp"
#include "svm-wrapper.hpp"
#include "hdf5_serialization.hpp"
#include "colormap.hpp"

#include <array>
#include <iostream>
#include <sstream>
#include <string>

#include <alps/hdf5.hpp>
#include <alps/utilities/fs/remove_extensions.hpp>

#include <boost/multi_array.hpp>


using sim_type = gauge_sim;
using kernel_t = svm::kernel::polynomial<2>;

void normalize_matrix (boost::multi_array<double,2> & mat) {
    double absmax = 0.;
    for (size_t i = 0; i < mat.shape()[0]; ++i)
        for (size_t j = 0; j < mat.shape()[0]; ++j)
            if (absmax < std::abs(mat[i][j]))
                absmax = std::abs(mat[i][j]);
    absmax = 1./absmax;
    for (size_t i = 0; i < mat.shape()[0]; ++i)
        for (size_t j = 0; j < mat.shape()[0]; ++j)
            mat[i][j] *= absmax;
}

template <typename Palette>
void write_matrix (boost::multi_array<double,2> const& mat, std::string basename,
                   Palette const& pal) {
    /* PPM output */ {
        std::ofstream ppm(basename + ".ppm", std::ios::binary);
        std::string header = [&] {
            std::stringstream ss;
            ss << "P6\n" << mat.shape()[0] << ' ' << mat.shape()[1] << '\n'
            << size_t(Palette::color_type::depth()) << '\n';
            return ss.str();
        } ();
        ppm.write(header.c_str(), header.size());
        for (int i = mat.shape()[0] - 1; i >= 0; --i) {
            for (size_t j = 0; j < mat.shape()[1]; ++j) {
                auto pix = pal(mat[i][j]);
                pix.write(ppm);
            }
        }
    }
    /* TXT output */ {
        std::ofstream txt(basename + ".txt");
        for (size_t i = 0; i < mat.shape()[0]; ++i) {
            for (size_t j = 0; j < mat.shape()[1]; ++j) {
                txt << mat[i][j] << '\t';
            }
            txt << '\n';
        }
    }
}

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

        std::unique_ptr<config_policy> confpol =
            sim_type::config_policy_from_parameters(parameters);
        {
            auto rearranged_coeffs = confpol->rearrange_by_component(coeffs);
            normalize_matrix(rearranged_coeffs);
            write_matrix(rearranged_coeffs,
                         alps::fs::remove_extensions(arname) + ".coeffs",
                         color::palettes.at("rdbu").rescale(-1, 1));
        }
        {
            auto block_structure = confpol->block_structure(coeffs);
            normalize_matrix(block_structure);
            write_matrix(block_structure,
                         alps::fs::remove_extensions(arname) + ".blocks",
                         color::palettes.at("parula"));
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
