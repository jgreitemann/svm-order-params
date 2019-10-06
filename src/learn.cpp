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

#include <iostream>
#include <exception>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <argh.h>

#include <alps/hdf5.hpp>
#include <alps/params.hpp>

#include <svm/svm.hpp>
#include <svm/serialization/hdf5.hpp>

#include <tksvm/config_sim_base.hpp>
#include <tksvm/phase_space/classifier.hpp>
#include <tksvm/sim_adapters/test_adapter.hpp>
#include <tksvm/utilities/mpi/mpi.hpp>

#ifdef CONFIG_MAPPING_LAZY
#include <tksvm/sim_adapters/procrastination_adapter.hpp>
using sim_type = tksvm::procrastination_adapter<tksvm::sim_base>;
#else
#include <tksvm/sim_adapters/training_adapter.hpp>
using sim_type = tksvm::training_adapter<tksvm::sim_base>;
#endif


using namespace tksvm;

using kernel_t = typename sim_type::kernel_t;
using label_t = typename sim_type::phase_label;
using phase_point = typename sim_type::phase_point;
using model_t = svm::model<kernel_t, label_t>;
using problem_t = typename model_t::problem_t;

int main(int argc, char** argv)
{
    mpi::environment env(argc, argv, mpi::environment::threading::multiple);
    try {
        // Creates the parameters for the simulation
        // If an hdf5 file is supplied, reads the parameters there
        std::cout << "Initializing parameters..." << std::endl;
        alps::params parameters(argc, argv);
        argh::parser cmdl(argc, argv, argh::parser::SINGLE_DASH_IS_MULTIFLAG);

        // define parameters
        sim_type::define_parameters(parameters);
        test_adapter<sim_base>::define_test_parameters(parameters);
        if (!parameters.is_restored()) {
            parameters.define<double>("nu", 0.5, "nu_SVC regularization parameter");
            parameters.define<size_t>("progress_interval", 3,
                                      "time in sec between progress reports");
        }

        if (parameters.help_requested(std::cout) ||
            parameters.has_missing(std::cout)) {
            return 1;
        }

        problem_t prob(0);
        auto classifier = phase_space::classifier::from_parameters<phase_point>(
            parameters, "classifier.");

        auto process_archive = [&](alps::hdf5::archive & cp) {
            if (!cp.is_open())
                throw std::runtime_error(
                    "Unable to open archive: " + cp.get_filename());
            int n_clones;
            cp["simulation/n_clones"] >> n_clones;

            phase_point first_point;
            for (int tid = 0; tid < n_clones; ++tid) {
                sim_type sim(parameters, tid);

                std::string checkpoint_path = [&] {
                    std::stringstream ss;
                    ss << "simulation/clones/" << tid;
                    return ss.str();
                } ();

                std::cout << "Restoring samples from " << cp.get_filename()
                        << " (clone " << tid << ")"
                        << std::endl;
                cp[checkpoint_path] >> sim;

                auto valid = [size = classifier->size()](label_t const& l) {
                    return size_t(l) <= size;
                };
                if (prob.dim() == 0) {
                    auto surrendered_problem = sim.surrender_problem();
                    first_point = surrendered_problem[0].second;
                    prob = problem_t(std::move(surrendered_problem),
                        classifier->get_functor(),
                        valid);
                } else {
                    prob.append_problem(sim.surrender_problem(),
                        classifier->get_functor(),
                        valid);
                }
            }
            return first_point;
        };

        phase_point first_point;
        if (parameters.is_restored()) {
            std::string checkpoint_file = parameters["checkpoint"].as<std::string>();
            alps::hdf5::archive cp(checkpoint_file, "r");
            first_point = process_archive(cp);
        } else {
            std::cerr
            << "The *-learn program no longer samples configurations but only "
            << "classifies samples and performs the actual SVM optimization.\n"
            << "Launch the *-sample program with the INI parameter file, then "
            << "provide the resulting .clone.h5 file as argument to *-learn.\n";
            return 1;
        }

        auto merge_is = cmdl({"--merge", "-m"});
        for (std::string name; std::getline(merge_is, name, ':');) {
            alps::hdf5::archive cp(name, "r");
            process_archive(cp);
        }

        if (cmdl[{"-i", "--infinite-temperature"}]) {
            training_adapter<sim_base> sim(parameters, 0);
            sim.update_phase_point(first_point);

            size_t N_samples = parameters["sweep.samples"].as<size_t>();
            for (size_t i = 0; i < N_samples; ++i) {
                sim.sample_config(sim.random_configuration(), phase_point{});
            }
            prob.append_problem(sim.surrender_problem(), [&](phase_point) {
                return classifier->infinity_label();
            });
        }

        /* print label statistics */ {
            std::map<label_t, size_t> label_stat;
            label_t l;
            for (size_t i = 0; i < prob.size(); ++i) {
                std::tie(std::ignore, l) = prob[i];
                auto it = label_stat.find(l);
                if (it != label_stat.end())
                    ++(it->second);
                else
                    label_stat.insert({l, 1});
            }
            std::cout << "\nLabel statistics:\n";
            for (auto const& p : label_stat) {
                std::cout << classifier->name(p.first) << ": " << p.second << '\n';
            }
            std::cout << std::endl;
        }

        if (!cmdl["--statistics-only"]) {
            // create the model
            svm::parameters<kernel_t> kernel_params(parameters["nu"].as<double>(),
                                                    svm::machine_type::NU_SVC);
            std::cout << "Creating SVM model..."
                      << " (nu = " << parameters["nu"].as<double>()
                      << ", total samples = " << prob.size() << ')'
                      << std::endl;
            model_t model(std::move(prob), kernel_params);

            // set up serializer
            svm::serialization::model_serializer<svm::hdf5_tag, model_t> serial(model);

            // Saving to the output file
            std::string output_file = parameters["outputfile"];
            alps::hdf5::archive ar(output_file, "w");
            ar["/parameters"] << parameters;
            ar["/model"] << serial;
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cout << "Exception caught: " << exc.what() << std::endl;
        return 2;
    }
}
