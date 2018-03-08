#include "gauge.hpp"
#include "results.hpp"
#include "svm-wrapper.hpp"
#include "hdf5_serialization.hpp"
#include "colormap.hpp"
#include "argh.h"
#include "filesystem.hpp"

#include <array>
#include <iostream>
#include <sstream>
#include <string>

#include <alps/hdf5.hpp>

#include <boost/multi_array.hpp>


using sim_type = gauge_sim;
using kernel_t = svm::kernel::polynomial<2>;

double normalize_matrix (boost::multi_array<double,2> & mat) {
    double absmax = 0.;
    for (size_t i = 0; i < mat.shape()[0]; ++i)
        for (size_t j = 0; j < mat.shape()[0]; ++j)
            if (absmax < std::abs(mat[i][j]))
                absmax = std::abs(mat[i][j]);
    for (size_t i = 0; i < mat.shape()[0]; ++i)
        for (size_t j = 0; j < mat.shape()[0]; ++j)
            mat[i][j] /= absmax;
    return absmax;
}

template <typename Palette>
void write_matrix (boost::multi_array<double,2> const& mat, std::string basename,
                   Palette const& pal) {
    /* PPM output */ {
        std::array<size_t, 2> shape {mat.shape()[0], mat.shape()[1]};
        typedef boost::multi_array_types::index_range range;
        boost::multi_array<double,2> flat(mat[boost::indices[range(shape[0]-1, -1, -1)][range(0, shape[1])]]);
        flat.reshape(std::array<size_t,2>{1, flat.num_elements()});
        auto pixit = itadpt::map_iterator(flat[0].begin(), pal);
        auto pmap = color::pixmap<decltype(pixit)>(pixit, shape);

        std::ofstream ppm(basename + "." + pmap.file_extension(), std::ios::binary);
        pmap.write_binary(ppm);
    }
    /* TXT output */ {
        std::ofstream txt(basename + ".txt");
        for (auto row_it = mat.rbegin(); row_it != mat.rend(); ++row_it) {
            for (double elem : *row_it) {
                txt << elem << '\t';
            }
            txt << '\n';
        }
    }
}

int main(int argc, char** argv) {
    try {
        argh::parser cmdl(argv, argh::parser::SINGLE_DASH_IS_MULTIFLAG);
        alps::params parameters = [&] {
            if (cmdl[1].empty())
                return alps::params(argc, argv);
            std::string pseudo_args[] = {cmdl[0], cmdl[1]};
            if (cmdl[{"-h", "--help"}])
                pseudo_args[1] = "--help";
            char const * pseudo_argv[] = {pseudo_args[0].c_str(), pseudo_args[1].c_str()};
            return alps::params(2, pseudo_argv);
        } ();
        sim_type::define_parameters(parameters);

        if (parameters.help_requested(std::cout) ||
            parameters.has_missing(std::cout)) {
            return 1;
        }

        std::string arname = parameters.get_archive_name();
        bool verbose = cmdl[{"-v", "--verbose"}];
        auto log_msg = [verbose] (std::string const& msg) {
            if (verbose)
                std::cout << msg << std::endl;
        };

        log_msg("Reading model...");
        svm::model<kernel_t> model;
        {
            alps::hdf5::archive ar(arname, "r");
            svm::model_serializer<svm::hdf5_tag, svm::model<kernel_t>> serial(model);
            ar["model"] >> serial;
        }

        svm::tensor_introspector<kernel_t, 2> coeff(model);

        log_msg("Allocating coeffs...");
        boost::multi_array<double,2> coeffs(boost::extents[model.dim()][model.dim()]);
        log_msg("Filling coeffs...");
#pragma omp parallel for
        for (size_t i = 0; i < model.dim(); ++i) {
            for (size_t j = 0; j < model.dim(); ++j) {
                coeffs[i][j] = coeff.tensor({i, j});
            }
        }

        std::unique_ptr<config_policy> confpol =
            sim_type::config_policy_from_parameters(parameters, cmdl[{"-u", "--unsymmetrize"}]);
        if (!cmdl[{"-b", "--blocks-only"}]) {
            log_msg("Rearranging coeffs...");
            auto rearranged_coeffs = confpol->rearrange_by_component(coeffs);
            log_msg("Normalizing coeffs...");
            normalize_matrix(rearranged_coeffs);
            log_msg("Writing coeffs...");
            write_matrix(rearranged_coeffs,
                         replace_extension(arname, ".coeffs"),
                         color::palettes.at("rdbu").rescale(-1, 1));
            if (cmdl[{"-e", "--exact"}] || cmdl[{"-d", "--diff"}]) {
                parameters["symmetrized"] = false;
                auto cpol = sim_type::config_policy_from_parameters(parameters, false);
                try {
                    auto exact = cpol->rearrange_by_component(
                        exact_tensor.at(parameters["gauge_group"]).get(cpol));
                    normalize_matrix(exact);
                    if (cmdl[{"-e", "--exact"}]) {
                        write_matrix(exact,
                                     replace_extension(arname, ".exact"),
                                     color::palettes.at("rdbu").rescale(-1, 1));
                    }

                    if (cmdl[{"-d", "--diff"}]) {
                        block_reduction::norm<2> norm_diff, norm_exact;
                        auto it_row_exact = exact.begin();
                        for (auto row : rearranged_coeffs) {
                            auto it_elem_exact = it_row_exact->begin();
                            for (auto & elem : row) {
                                elem -= *it_elem_exact;
                                norm_diff += elem;
                                norm_exact += *it_elem_exact;
                                ++it_elem_exact;
                            }
                            ++it_row_exact;
                        }
                        std::cout << "relative scale of difference tensor: "
                                  << normalize_matrix(rearranged_coeffs)
                                  << std::endl;
                        write_matrix(rearranged_coeffs,
                                     replace_extension(arname, ".diff"),
                                     color::palettes.at("rdbu").rescale(-1, 1));
                        std::cout << "deviation metric: " << double(norm_diff) << '\n'
                                  << "total Frobenius norm: " << double(norm_exact) << '\n'
                                  << "relative deviation: " << double(norm_diff)/double(norm_exact)
                                  << std::endl;
                        auto nSV = model.nSV();
                        std::ofstream os (replace_extension(arname, ".dev.txt"));
                        os << parameters["length"].as<size_t>() << '\t'
                           << parameters["temp_crit"].as<double>() << '\t'
                           << (parameters["sweep_unit"].as<size_t>()
                               * parameters["total_sweeps"].as<size_t>()
                               / parameters["N_sample"].as<size_t>()) << '\t'
                           << (parameters["sweep_unit"].as<size_t>()
                               * parameters["thermalization_sweeps"].as<size_t>()) << '\t'
                           << parameters["nu"].as<double>() << '\t'
                           << nSV.first << '\t' << nSV.second << '\t'
                           << double(norm_diff) << '\t'
                           << double(norm_exact) << '\t'
                           << double(norm_diff) / double(norm_exact) << '\n';
                    }
                } catch (std::out_of_range const& e) {
                    std::cerr << "No exact solution know for symmetry \""
                              << parameters["gauge_group"]
                              << "\" despite --exact flag given."
                              << std::endl;
                }
            }
        }
        {
            log_msg("Block structure...");
            auto block_structure = confpol->block_structure(coeffs);
            normalize_matrix(block_structure);
            write_matrix(block_structure,
                         replace_extension(arname, ".blocks"),
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
