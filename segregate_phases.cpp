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

#include "phase_space_policy.hpp"
#include "svm-wrapper.hpp"
#include "hdf5_serialization.hpp"
#include "argh.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>

#include <Eigen/Eigenvalues>

#include "config_sim_base.hpp"

using phase_point = typename sim_base::phase_point;
using kernel_t = svm::kernel::polynomial<2>;
using label_t = typename sim_base::phase_label;
using model_t = svm::model<kernel_t, label_t>;

int main(int argc, char** argv)
{
    argh::parser cmdl({"r", "rhoc", "p", "phase"});
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

    double rhoc;
    if (!(cmdl({"-r", "--rhoc"}) >> rhoc))
        rhoc = 1.75;

    std::string arname = parameters.get_archive_name();
    bool verbose = cmdl[{"-v", "--verbose"}];
    auto log_msg = [verbose] (std::string const& msg) {
        if (verbose)
            std::cout << msg << std::endl;
    };

    log_msg("Reading model...");
    model_t model;
    {
        alps::hdf5::archive ar(arname, "r");
        svm::model_serializer<svm::hdf5_tag, model_t> serial(model);
        ar["model"] >> serial;
    }

    std::map<label_t, phase_point> phase_points;
    std::map<label_t, size_t> index_map;
    std::vector<label_t> labels;
    {
        using classifier_t = typename sim_base::phase_classifier;
        auto grid_sweep = phase_space::sweep::from_parameters<phase_point>(parameters);
        classifier_t classifier(parameters);
        phase_point p;
        std::ofstream os("vertices.txt");
        std::mt19937 dummy_rng(42);
        for (size_t i = 0; i < grid_sweep->size(); ++i) {
            grid_sweep->yield(p, dummy_rng);
            auto l = classifier(p);
            phase_points[l] = p;
            index_map[l] = i;
            labels.push_back(l);
            os << l << '\t';
            std::copy(p.begin(), p.end(), std::ostream_iterator<double>{os, "\t"});
            os << '\n';
        }
    }

    using matrix_t = Eigen::MatrixXd;
    matrix_t L(phase_points.size(), phase_points.size());
    {
        std::ofstream os("rho.txt");
        std::ofstream os2("edges.txt");
        for (auto const& transition : model.classifiers()) {
            auto labels = transition.labels();
            size_t i = index_map[labels.first], j = index_map[labels.second];
            double rho = std::abs(std::abs(transition.rho()) - 1);

            if (rho > rhoc) {
                std::copy(phase_points[labels.first].begin(),
                          phase_points[labels.first].end(),
                          std::ostream_iterator<double> {os2, "\t"});
                os2 << '\n';
                std::copy(phase_points[labels.second].begin(),
                          phase_points[labels.second].end(),
                          std::ostream_iterator<double> {os2, "\t"});
                os2 << "\n\n";

                L(i,j) = -1;
                L(j,i) = -1;
                L(i,i) += 1;
                L(j,j) += 1;
            }

            // auto diag = phase_space::classifier::D2h_map.at("D2h");
            // bool is_transition = diag(phase_points[labels.first]) == diag(phase_points[labels.second]);
            // os << is_transition << '\t' << rho << std::endl;
            os << std::abs(transition.rho()) << '\n';
        }
    }

    auto eigen = Eigen::SelfAdjointEigenSolver<matrix_t>(L);

    std::vector<std::pair<size_t, double>> evals;
    evals.reserve(phase_points.size());
    for (size_t i = 0; i < phase_points.size(); ++i)
        evals.emplace_back(i, eigen.eigenvalues()(i));
    std::sort(evals.begin(), evals.end(),
              [](auto const& lhs, auto const& rhs) { return lhs.second < rhs.second; });

    size_t degen = 0;
    std::ofstream os("phases.txt");
    auto const& evecs = eigen.eigenvectors();
    for (size_t i = 0; i < phase_points.size(); ++i) {
        os << "# eval = " << evals[i].second << '\n';
        if (evals[i].second < 1e-10)
            ++degen;
        for (size_t j = 0; j < phase_points.size(); ++j) {
            auto const& p = phase_points[labels[j]];
            std::copy(p.begin(), p.end(),
                      std::ostream_iterator<double>{os, "\t"});
            os << evecs(j, evals[i].first) << '\n';
        }
        os << "\n\n";
    }

    std::cout << "Degeneracy of smallest eval: " << degen << std::endl;
}
