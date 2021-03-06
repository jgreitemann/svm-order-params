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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <argh.h>

#include <alps/hdf5.hpp>
#include <alps/params.hpp>

#include <Eigen/Eigenvalues>

#include <svm/svm.hpp>
#include <svm/serialization/hdf5.hpp>

#include <tksvm/config_sim_base.hpp>
#include <tksvm/phase_space/classifier.hpp>
#include <tksvm/phase_space/sweep.hpp>
#include <tksvm/phase_space/point/common.hpp>


using namespace tksvm;

using phase_point = typename sim_base::phase_point;
using kernel_t = svm::kernel::polynomial<2>;
    using label_t = typename phase_space::classifier::policy<phase_point>::label_type;
using model_t = svm::model<kernel_t, label_t>;

int main(int argc, char** argv)
{
    argh::parser cmdl({"r", "rhoc", "R", "radius", "m", "mask", "masked-value",
        "t", "threshold", "w", "weight"});
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
    sim_base::define_parameters(parameters);

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

    double radius;
    cmdl({"-R", "--radius"}, std::numeric_limits<double>::max()) >> radius;

    log_msg("Reading model...");
    model_t model;
    {
        alps::hdf5::archive ar(arname, "r");
        svm::serialization::model_serializer<svm::hdf5_tag, model_t> serial(model);
        ar["model"] >> serial;
    }

    log_msg("Calculating bias statistics...");
    double rho_std, median, hiqs;
    {
        double variance = 0;
        size_t n = 0;
        std::vector<double> rhos;
        for (auto const& transition : model.classifiers()) {
            ++n;
            double rho = std::abs(transition.rho());
            rhos.push_back(rho);
            variance = (1. * (n - 1) / n) * variance + pow(rho - 1., 2.) / n;
        }
        rho_std = sqrt(variance);

        std::sort(rhos.begin(), rhos.end());
        median = rhos[rhos.size() / 2];
        hiqs = 0.5 * (rhos[rhos.size() * 3 / 4] - rhos[rhos.size() / 4]);
    }
    std::cout << "RMS bias deviation from unity: " << rho_std << '\n'
              << "Median bias: " << median << '\n'
              << "Half interquartile spacing: " << hiqs << '\n';

    auto weight = [&]() -> std::function<double(double)> {
        double rhoc;
        std::string weight_name = cmdl({"-w", "--weight"}, "box").str();
        if (weight_name == "box") {
            cmdl({"-r", "--rhoc"}, 1.) >> rhoc;
            return [rhoc](double rho) {
                return std::abs(std::abs(rho) - 1.) > rhoc;
            };
        } else if (weight_name == "gaussian") {
            cmdl({"-r", "--rhoc"}, rho_std) >> rhoc;
            return [rhoc](double rho) {
                return 1. - exp(-0.5 * pow((std::abs(rho) - 1.) / rhoc, 2.));
            };
        } else if (weight_name == "lorentzian") {
            cmdl({"-r", "--rhoc"}, hiqs) >> rhoc;
            return [&, gamma_sq = rhoc * rhoc](double rho) {
                return 1. - gamma_sq / (pow(std::abs(rho) - 1., 2.) + gamma_sq);
            };
        } else {
            throw std::runtime_error("unknown weight function: " + weight_name);
        }
    }();

    log_msg("Collecting phase space points...");
    std::map<label_t, phase_point> phase_points;
    std::map<label_t, size_t> index_map;
    std::vector<label_t> labels;
    size_t graph_dim = [&] {
        auto unique_phase_points = [&] {
            auto sweep_pol = phase_space::sweep::from_parameters<phase_point>(
                parameters, "sweep.");
            std::vector<phase_point> points;
            phase_space::point::distance<phase_point> dist;
            auto process_sweep = [&](alps::params const& parameters) {
                auto sweep_pol = phase_space::sweep::from_parameters<phase_point>(
                    parameters, "sweep.");
                std::mt19937 rng{parameters["SEED"].as<std::mt19937::result_type>()};
                phase_point p;
                for (size_t i = 0; i < sweep_pol->size(); ++i) {
                    sweep_pol->yield(p, rng);
                    double min_dist = std::numeric_limits<double>::max();
                    for (auto const& pp : points)
                        min_dist = std::min(min_dist, dist(p, pp));
                    if (min_dist > 1e-5)
                        points.push_back(p);
                }
            };
            process_sweep(parameters);

            // push points from sweep policies of merged clones
            alps::hdf5::archive ar(alps::origin_name(parameters), "r");
            std::string path;
            for (size_t i = 0; ar.is_group(path = "/merged_parameters/" + std::to_string(i)); ++i) {
                alps::params merged_params;
                ar[path] >> merged_params;
                process_sweep(merged_params);
            }

            return points;
        }();

        // read mask if specified
        std::vector<int> mask = [&] {
            std::string mask_string;
            if (cmdl({"-m", "--mask"}) >> mask_string) {
                std::regex re{"[^:,]+"};
                auto parse_mask = [&](std::string mask_name) {
                    log_msg("Applying mask: " + mask_name);
                    std::ifstream is{mask_name};
                    if (!is) {
                        std::cerr << "unable to open mask file: " << mask_name
                                  << "\tskipping...\n";
                        return std::vector<int>(unique_phase_points.size(), 1);
                    }
                    phase_point p;
                    std::vector<int> mask;
                    while (true) {
                        for (double & x : p)
                            if (!(is >> x))
                                break;
                        if (!is)
                            break;
                        mask.emplace_back();
                        is >> mask.back();
                    }
                    if (mask.size() != unique_phase_points.size())
                        throw std::runtime_error("inconsistent mask size");
                    return mask;
                };

                std::smatch sm;
                std::regex_search(mask_string, sm, re);
                std::vector<int> combined_mask = parse_mask(sm.str());
                while (mask_string = sm.suffix(), std::regex_search(mask_string, sm, re)) {
                    auto other_mask = parse_mask(sm.str());
                    std::transform(combined_mask.begin(), combined_mask.end(),
                        other_mask.begin(), combined_mask.begin(),
                        std::multiplies<>{});
                }
                return combined_mask;
            }
            return std::vector<int>(unique_phase_points.size(), 1);
        }();

        auto classifier = phase_space::classifier::from_parameters<phase_point>(
            parameters, "classifier.");
        auto model_labels = model.labels();
        std::ofstream os("vertices.txt");
        size_t i = 0, j = 0;
        for (auto const& p : unique_phase_points) {
            auto l = (*classifier)(p);
            phase_points[l] = p;
            if (!mask[i++])
                continue;
            if (std::find(model_labels.begin(), model_labels.end(), l)
                    == model_labels.end())
                continue;
            labels.push_back(l);
            index_map[l] = j++;
            os << l << '\t';
            std::copy(p.begin(), p.end(), std::ostream_iterator<double>{os, "\t"});
            os << '\n';
        }
        return j;
    }();

    log_msg("Accessing auxiliary graphs to combine...");
    std::vector<std::vector<double>> aux_weights;
    auto const& args = cmdl.pos_args();
    for (auto it = std::next(args.begin(), 2); it != args.end(); ++it) {
        double rho, w;
        std::ifstream is{it->c_str()};
        if (is) {
            std::cout << "Reading auxiliary graph: " << it->c_str() << '\n';
        } else {
            std::cerr << "Could not open file: " << it->c_str()
                      << "\tskipping...\n";
            continue;
        }
        aux_weights.emplace_back();
        while (is >> rho >> w)
            aux_weights.back().push_back(w);
        if (aux_weights.back().size() != model.nr_classifiers())
            throw std::runtime_error("inconsistent number of graph edges in "
                + *it);
    }

    log_msg("Constructing graph...");
    using matrix_t = Eigen::MatrixXd;
    matrix_t L = matrix_t::Zero(graph_dim, graph_dim);
    {
        // get auxiliary weights iterators
        using iter_t = typename std::vector<double>::const_iterator;
        std::vector<iter_t> aux_iters;
        std::transform(aux_weights.begin(), aux_weights.end(),
            std::back_inserter(aux_iters),
            std::mem_fn(&std::vector<double>::cbegin));

        std::ofstream os("rho.txt");
        std::ofstream os2("edges.txt");
        phase_space::point::distance<phase_point> dist{};
        for (auto const& transition : model.classifiers()) {
            auto labels = transition.labels();
            if (size_t(labels.first) == phase_points.size()
                || size_t(labels.second) == phase_points.size())
                continue;

            double w = weight(transition.rho());

            // combine weights from auxiliary graphs
            for (auto & it : aux_iters)
                w *= *(it++);

            os << std::abs(transition.rho()) << '\t' << w << '\n';

            if (index_map.find(labels.first) == index_map.end()
                || index_map.find(labels.second) == index_map.end())
            {
                continue;
            }

            if (dist(phase_points[labels.first], phase_points[labels.second]) > radius)
                continue;

            size_t i = index_map[labels.first], j = index_map[labels.second];
            if (w > 0) {
                std::copy(phase_points[labels.first].begin(),
                    phase_points[labels.first].end(),
                    std::ostream_iterator<double> {os2, "\t"});
                std::copy(phase_points[labels.second].begin(),
                    phase_points[labels.second].end(),
                    std::ostream_iterator<double> {os2, "\t"});
                os2 << w << '\n';
            }

            L(i,j) = -w;
            L(j,i) = -w;
            L(i,i) += w;
            L(j,j) += w;
        }
    }

    log_msg("Diagonalizing Laplacian...");
    auto eigen = Eigen::SelfAdjointEigenSolver<matrix_t>(L);
    auto const& evecs = eigen.eigenvectors();

    std::vector<std::pair<size_t, double>> evals;
    evals.reserve(phase_points.size());
    for (size_t i = 0; i < graph_dim; ++i)
        evals.emplace_back(i, eigen.eigenvalues()(i));
    std::sort(evals.begin(), evals.end(),
              [](auto const& lhs, auto const& rhs) { return lhs.second < rhs.second; });

    log_msg("Writing phases...");
    size_t degen = 0;
    {
        std::ofstream os("phases.txt");
        double masked_value;
        bool use_masked_value = bool(cmdl("--masked-value") >> masked_value);
        for (size_t i = 0; i < graph_dim; ++i) {
            os << "# eval = " << evals[i].second << '\n';
            if (evals[i].second < 1e-10)
                ++degen;
            label_t l;
            phase_point p;
            for (auto const& label_point_pair : phase_points) {
                std::tie(l, p) = label_point_pair;
                auto idx_it = index_map.find(l);
                if (idx_it == index_map.end()) {
                    if (use_masked_value) {
                        std::copy(p.begin(), p.end(),
                            std::ostream_iterator<double>{os, "\t"});
                        os << masked_value << '\n';
                    } else {
                        continue;
                    }
                } else {
                    std::copy(p.begin(), p.end(),
                        std::ostream_iterator<double>{os, "\t"});
                    os << evecs(idx_it->second, evals[i].first) << '\n';
                }
            }
            os << "\n\n";
        }
    }

    log_msg("Writing mask...");
    double threshold;
    if (cmdl({"-t", "--threshold"}) >> threshold) {
        bool invert_mask = cmdl["--invert-mask"];
        std::ofstream os("mask.txt");
        auto const& fiedler_vec = evecs.col(evals[degen].first);
        for (auto it = phase_points.begin(); it != phase_points.end(); ++it) {
            auto const& l = it->first;
            auto const& p = it->second;
            std::copy(p.begin(), p.end(),
                std::ostream_iterator<double>{os, "\t"});
            auto idx_it = index_map.find(l);
            if (idx_it == index_map.end())
                os << 0 << '\n';
            else
                os << ((fiedler_vec(idx_it->second) >= threshold) ^ invert_mask)
                   << '\n';
        }
    }

    std::cout << "Degeneracy of smallest eval: " << degen << std::endl;
}
