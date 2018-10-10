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

#include "gauge.hpp"
#include "results.hpp"
#include "svm-wrapper.hpp"
#include "hdf5_serialization.hpp"
#include "colormap.hpp"
#include "argh.h"
#include "filesystem.hpp"
#include "contraction.hpp"
#include "matrix_output.hpp"

#include <array>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

#include <alps/hdf5.hpp>

#include <boost/multi_array.hpp>

#include <Eigen/SVD>


using sim_type = gauge_sim;
using kernel_t = svm::kernel::polynomial<2>;


int main(int argc, char** argv) {
    try {
        argh::parser cmdl;
        cmdl.add_params({"block", "t", "transition", "result"});
        cmdl.parse(argc, argv, argh::parser::SINGLE_DASH_IS_MULTIFLAG);
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
        bool verbose = cmdl[{"-v", "--verbose"}] || cmdl[{"-c", "--contraction-weights"}];
        auto log_msg = [verbose] (std::string const& msg) {
            if (verbose)
                std::cout << msg << std::endl;
        };

        log_msg("Reading model...");
        using model_t = svm::model<kernel_t, sim_type::phase_label>;
        using classifier_t = typename model_t::classifier_type;
        model_t model;
        {
            alps::hdf5::archive ar(arname, "r");
            svm::model_serializer<svm::hdf5_tag, model_t> serial(model);
            ar["model"] >> serial;
        }

        auto treat_transition = [&] (classifier_t const& classifier,
                                     std::string const& basename)
        {
            auto coeff = svm::tensor_introspect<2>(classifier);

            std::unique_ptr<config_policy> confpol =
                sim_type::config_policy_from_parameters(parameters,
                                                        cmdl[{"-u", "--unsymmetrize"}]);

            auto contractions = get_contractions(confpol->rank());
            auto block_inds = confpol->all_block_indices();
            using block_ind_t = decltype(block_inds)::value_type;

            std::string block_name;
            if (cmdl("block") >> block_name) {

                std::regex block_regex([&] {
                        std::stringstream ss;
                        ss << "[^lmn]*([lmn]{"
                        << confpol->rank()
                        << "})[,.;:| ]([lmn]{"
                        << confpol->rank()
                        << "})[^lmn]*";
                        return ss.str();
                    } ());
                std::smatch match;
                if (!std::regex_match(block_name, match, block_regex) || match.size() != 3) {
                    throw std::runtime_error("could not parse block: '"
                                             + block_name + "'");
                }

                auto block_ind_to_string = [] (block_ind_t const& b) {
                    std::stringstream ss;
                    ss << block_indices_t { b.first };
                    return ss.str();
                };

                auto bi_it = std::find_if(block_inds.begin(), block_inds.end(),
                                          [&] (block_ind_t const& b) {
                                              return block_ind_to_string(b) == match[1];
                                          });
                auto bj_it = std::find_if(block_inds.begin(), block_inds.end(),
                                          [&] (block_ind_t const& b) {
                                              return block_ind_to_string(b) == match[2];
                                          });
                if (bi_it == block_inds.end() || bj_it == block_inds.end())
                    throw std::runtime_error([&] {
                            std::stringstream ss;
                            ss << "block [" << match[1] << ";" << match[2]
                               << "] not found";
                            return ss.str();
                        } ());
                auto const& bi = *bi_it;
                auto const& bj = *bj_it;

                log_msg("Allocating " + block_str(bi.first, bj.first) + " block coeffs...");
                boost::multi_array<double,2> coeffs;

                if (!cmdl[{"-r", "--raw"}]) {
                    log_msg("Calculating and rearranging coeffs...");
                    auto rearranged_coeffs = confpol->rearrange(coeff, bi.first, bj.first);

                    // in single-block mode, contraction analysis operates on the
                    // rearranged (unsymmetrized) full block, *including* redundancies.
                    if (cmdl[{"-s", "--remove-self-contractions"}]
                        || cmdl[{"-c", "--contraction-weights"}])
                    {
                        log_msg("Analyzing contractions...");

                        using Container = typename config_policy::config_array;
                        using ElementPolicy = element_policy::mono;
                        using LatticePolicy = lattice::single<ElementPolicy,
                                                              Container>;
                        using confpol_t = gauge_config_policy<LatticePolicy,
                                                              symmetry_policy::none>;
                        confpol_t block_confpol(confpol->rank(), ElementPolicy{}, false);
                        auto block = block_confpol.all_block_indices().begin()->second;

                        auto a = contraction_matrix(contractions,
                                                    block,
                                                    block);

                        // collect the RHS vector
                        contraction_vector_t b(block.size() * block.size());
                        for (size_t i = 0; i < block.size(); ++i)
                            for (size_t j = 0; j < block.size(); ++j)
                                b(i * block.size() + j) = rearranged_coeffs[i][j];

                        Eigen::VectorXd x = a.bdcSvd(Eigen::ComputeThinU
                                                     | Eigen::ComputeThinV).solve(b);

                        std::stringstream contr_ss;
                        if (cmdl[{"-c", "--contraction-weights"}]) {
                            for (size_t i = 0; i < contractions.size(); ++i) {
                                contr_ss
                                    << contractions[i] << '\t'
                                    << (contractions[i].is_self_contraction()
                                        ? "self" : "outer")
                                    << '\t' << x[i] << '\n';

                            }
                        }
                        if (cmdl[{"-s", "--remove-self-contractions"}]) {
                            for (size_t i = 0; i < contractions.size(); ++i) {
                                if (!contractions[i].is_self_contraction())
                                    x[i] = 0.;
                            }

                            b = a * x;

                            // subtract self-contractions from coeffs
                            for (size_t i = 0; i < block.size(); ++i)
                                for (size_t j = 0; j < block.size(); ++j)
                                    rearranged_coeffs[i][j] -= b(i * block.size() + j);

                        }
                        if (cmdl[{"-c", "--contraction-weights"}])
    #pragma omp critical
                            log_msg(contr_ss.str());
                    }

                    log_msg("Normalizing coeffs...");
                    normalize_matrix(rearranged_coeffs);
                    log_msg("Writing coeffs...");
                    write_matrix(rearranged_coeffs,
                                 replace_extension(basename,
                                                   "."
                                                   + block_str(bi.first, bj.first)
                                                   + ".coeffs"),
                                 color::palettes.at("rdwhbu").rescale(-1, 1));
                } else {
                    coeffs.resize(boost::extents[bi.second.size()][bj.second.size()]);
                    log_msg("Filling coeffs...");
    #pragma omp parallel for
                    for (size_t i = 0; i < coeffs.shape()[0]; ++i) {
                        for (size_t j = 0; j < coeffs.shape()[1]; ++j) {
                            coeffs[i][j] = coeff.tensor({bi.second[i].first,
                                                         bj.second[j].first});
                        }
                    }
                    log_msg("Normalizing coeffs...");
                    normalize_matrix(coeffs);
                    log_msg("Writing coeffs...");
                    write_matrix(coeffs,
                                 replace_extension(basename,
                                                   "."
                                                   + block_str(bi.first, bj.first)
                                                   + ".coeffs"),
                                 color::palettes.at("rdwhbu").rescale(-1, 1));
                }

            } else {

                log_msg("Allocating coeffs...");
                boost::multi_array<double,2> coeffs(boost::extents[model.dim()][model.dim()]);
                log_msg("Filling coeffs...");
    #pragma omp parallel for
                for (size_t i = 0; i < model.dim(); ++i) {
                    for (size_t j = 0; j < model.dim(); ++j) {
                        coeffs[i][j] = coeff.tensor({i, j});
                    }
                }
                std::vector<block_ind_t> block_inds_vec(block_inds.begin(),
                                                        block_inds.end());
                size_t n_blocks = pow(block_inds.size(), 2);
                size_t i_block = 0;

                if (cmdl[{"-s", "--remove-self-contractions"}]
                    || cmdl[{"-c", "--contraction-weights"}])
                {
                    log_msg("Analyzing contractions...");
    #pragma omp parallel for
                    for (size_t bii = 0; bii < block_inds_vec.size(); ++bii) {
                        auto const& bi = block_inds_vec[bii];
                        for (size_t bjj = 0; bjj < block_inds_vec.size(); ++bjj) {
                            auto const& bj = block_inds_vec[bjj];
                            std::stringstream contr_ss;

                            auto a = contraction_matrix(contractions,
                                                        bi.second,
                                                        bj.second);

                            // crop the RHS vector
                            contraction_vector_t b(bi.second.size() * bj.second.size());
                            for (size_t i = 0; i < bi.second.size(); ++i)
                                for (size_t j = 0; j < bj.second.size(); ++j)
                                    b(i * bj.second.size() + j)
                                        = coeffs[bi.second[i].first][bj.second[j].first];

                            Eigen::VectorXd x = a.bdcSvd(Eigen::ComputeThinU
                                                         | Eigen::ComputeThinV).solve(b);

                            if (cmdl[{"-c", "--contraction-weights"}]) {
                                for (size_t i = 0; i < contractions.size(); ++i) {
                                    contr_ss
                                        << contractions[i] << '\t'
                                        << (contractions[i].is_self_contraction()
                                            ? "self" : "outer")
                                        << '\t' << x[i] << '\n';

                                }
                            }
                            if (cmdl[{"-s", "--remove-self-contractions"}]) {
                                for (size_t i = 0; i < contractions.size(); ++i) {
                                    if (!contractions[i].is_self_contraction())
                                        x[i] = 0.;
                                }

                                b = a * x;

                                // subtract self-contractions from coeffs
                                for (size_t i = 0; i < bi.second.size(); ++i)
                                    for (size_t j = 0; j < bj.second.size(); ++j)
                                        coeffs[bi.second[i].first][bj.second[j].first]
                                            -= b(i * bj.second.size() + j);

                            }
    #pragma omp critical
                            {
                                std::stringstream ss;
                                ++i_block;
                                ss << "Block " << block_str(bi.first, bj.first)
                                   << " (" << i_block << " / " << n_blocks << ")";
                                log_msg(ss.str());
                                if (cmdl[{"-c", "--contraction-weights"}])
                                    log_msg(contr_ss.str());
                            }
                        }
                    }
                }

                if (!cmdl[{"-b", "--blocks-only"}] && !cmdl[{"-r", "--raw"}]) {
                    log_msg("Rearranging coeffs...");
                    auto rearranged_coeffs = confpol->rearrange(coeffs);
                    log_msg("Normalizing coeffs...");
                    normalize_matrix(rearranged_coeffs);
                    log_msg("Writing coeffs...");
                    write_matrix(rearranged_coeffs,
                                 replace_extension(basename, ".coeffs"),
                                 color::palettes.at("rdwhbu").rescale(-1, 1));
                    if (cmdl[{"-e", "--exact"}] || cmdl[{"-d", "--diff"}]) {
                        alps::params nosymm_params(parameters);
                        nosymm_params["symmetrized"] = false;
                        auto cpol = sim_type::config_policy_from_parameters(nosymm_params, false);
                        std::string result_name = parameters["gauge_group"];
                        cmdl("--result") >> result_name;
                        try {
                            auto exact = cpol->rearrange(
                                results::exact_tensor.at(result_name).get(cpol));
                            normalize_matrix(exact);
                            if (cmdl[{"-e", "--exact"}]) {
                                write_matrix(exact,
                                             replace_extension(basename, ".exact"),
                                             color::palettes.at("rdwhbu").rescale(-1, 1));
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
                                            replace_extension(basename, ".diff"),
                                            color::palettes.at("rdwhbu").rescale(-1, 1));
                                std::cout << "deviation metric: "
                                          << double(norm_diff) << '\n'
                                          << "total Frobenius norm: "
                                          << double(norm_exact) << '\n'
                                          << "relative deviation: "
                                          << double(norm_diff)/double(norm_exact)
                                          << std::endl;
                                std::ofstream os (replace_extension(basename, ".dev.txt"));
                                os << parameters["length"].as<size_t>() << '\t'
                                   << (parameters["sweep_unit"].as<size_t>()
                                       * parameters["total_sweeps"].as<size_t>()
                                       / parameters["sweep.samples"].as<size_t>()) << '\t'
                                   << (parameters["sweep_unit"].as<size_t>()
                                       * parameters["thermalization_sweeps"].as<size_t>()) << '\t'
                                   << parameters["nu"].as<double>() << '\t'
                                   << double(norm_diff) << '\t'
                                   << double(norm_exact) << '\t'
                                   << double(norm_diff) / double(norm_exact) << '\n';
                            }
                        } catch (std::out_of_range const& e) {
                            std::cerr << "No exact solution know for symmetry \""
                                      << result_name
                                      << std::endl;
                        }
                    }
                } else if (cmdl[{"-r", "--raw"}]) {
                    log_msg("Normalizing coeffs...");
                    normalize_matrix(coeffs);
                    log_msg("Writing coeffs...");
                    write_matrix(coeffs,
                                 replace_extension(basename, ".coeffs"),
                                 color::palettes.at("rdwhbu").rescale(-1, 1));
                }
                {
                    log_msg("Block structure...");
                    auto block_structure = confpol->block_structure(coeffs);
                    normalize_matrix(block_structure.first);
                    write_matrix(block_structure.first,
                                replace_extension(basename, ".blocks.norm2"),
                                color::palettes.at("whgnbu"));
                    normalize_matrix(block_structure.second);
                    write_matrix(block_structure.second,
                                replace_extension(basename, ".blocks.sum"),
                                color::palettes.at("rdwhbu").rescale(-1, 1));
                }
            }
        };

        // Determine requested transitions
        auto transitions = model.classifiers();
        size_t t;
        bool exclusive = bool(cmdl({"-t", "--transition"}) >> t);
        for (size_t k = 0; k < transitions.size(); ++k) {
            if (exclusive && t != k)
                continue;
            auto const& cl = transitions[k];
            std::cout << k << ":   " << cl.labels().first << " -- "
                      << cl.labels().second
                      << "\t rho = " << cl.rho() << std::endl;
            std::stringstream ss;
            ss << replace_extension(arname, "")
               << '-' << cl.labels().first
               << '-' << cl.labels().second;
            if (!cmdl[{"-l", "--list"}])
                treat_transition(cl, ss.str());
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
