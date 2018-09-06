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

#include <Eigen/LU>

#ifdef ISING
#include "ising.hpp"
using sim_base = ising_sim;
#else
#ifdef GAUGE
#include "gauge.hpp"
using sim_base = gauge_sim;
#else
#error Unknown model
#endif
#endif


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
        phase_space::sweep::grid<phase_point> grid_sweep(parameters);
        classifier_t classifier(parameters);
        phase_point p;
        for (size_t i = 0; i < grid_sweep.size(); ++i) {
            grid_sweep.yield(p);
            auto l = classifier(p);
            phase_points[l] = p;
            index_map[l] = i;
            labels.push_back(l);
        }
    }

    using matrix_t = Eigen::MatrixXd;
    matrix_t A(phase_points.size(), phase_points.size());
    for (auto p : index_map) {
        A(p.second, p.second) = 1.;
    }
    {
        std::ofstream os("rho.txt");
        std::ofstream os2("edges.txt");
        for (auto const& transition : model.classifiers()) {
            auto labels = transition.labels();
            size_t i = index_map[labels.first], j = index_map[labels.second];
            double rho = std::abs(transition.rho());
            double val = (rho > rhoc) ? 1. : 0.;

            if (rho > rhoc) {
                std::copy(phase_points[labels.first].begin(),
                          phase_points[labels.first].end(),
                          std::ostream_iterator<double> {os2, "\t"});
                os2 << '\n';
                std::copy(phase_points[labels.second].begin(),
                          phase_points[labels.second].end(),
                          std::ostream_iterator<double> {os2, "\t"});
                os2 << "\n\n";
            }

            auto diag = phase_space::classifier::D2h_map.at("D2h");
            bool is_transition = diag(phase_points[labels.first]) == diag(phase_points[labels.second]);
            os << is_transition << '\t' << rho << std::endl;

            A(i, j) = val;
            A(j, i) = val;
        }
    }

    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

    std::cout << svd.rank() << " phases found! (rhoc = " << rhoc
              << ")" << std::endl;

    auto const& S = svd.singularValues();
    auto const& U = svd.matrixU();
    auto const& V = svd.matrixV();

    size_t p;
    bool exclusive = bool(cmdl({"-p", "--phase"}) >> p);
    {
        std::ofstream os([&]() -> std::string {
                if (!exclusive)
                    return "phases.txt";
                std::stringstream ss;
                ss << "phase_" << p << ".txt";
                return ss.str();
            }());
        for (size_t i = 0; i < svd.rank(); ++i) {
            if (exclusive && i != p)
                continue;
            os << "# S = " << S(i) << '\n';
            for (size_t j = 0; j < phase_points.size(); ++j) {
                double val = U(j,i) * S(i) * V(j,i);
                auto const& pp = phase_points[labels[j]];
                std::copy(pp.begin(), pp.end(),
                          std::ostream_iterator<double>{os, "\t"});
                os << val << '\n';
            }
            os << "\n\n";
        }
    }


}
