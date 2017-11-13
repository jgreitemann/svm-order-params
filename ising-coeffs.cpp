#include "ising.hpp"
#include "svm-wrapper.hpp"
#include "hdf5_serialization.hpp"

#include <array>
#include <iostream>

#include <alps/hdf5.hpp>
#include <alps/utilities/fs/remove_extensions.hpp>

#include <boost/multi_array.hpp>


using kernel_t = svm::kernel::polynomial<2>;

int main(int argc, char** argv) {
    // Define the type for the simulation
    typedef ising_sim sim_type;

    try {
        alps::params parameters(argc, argv);
        sim_type::define_parameters(parameters);

        if (parameters.help_requested(std::cout) ||
            parameters.has_missing(std::cout)) {
            return 1;
        }

        std::string arname = parameters.get_archive_name();
        size_t length = parameters["length"].as<int>();

        alps::hdf5::archive ar(arname, "r");

        svm::model<kernel_t> model;
        svm::model_serializer<svm::hdf5_tag, svm::model<kernel_t>> serial(model);
        ar["model"] >> serial;

        svm::tensor_introspector<kernel_t, 2> coeff(model);

        boost::multi_array<double,2> C(boost::extents[length * length][1]);
        for (size_t x = 0; x < length * length; ++x) {
            C[x][0] = 0;
            for (size_t i = 0; i < length * length; ++i) {
                size_t j = (i + x) % (length * length);
                std::cout << '(' << i << ", " << j << ')' << std::endl;
                C[x][0] += coeff.tensor({i, j});
            }
            C[x][0] /= length * length;
        }
        C.reshape(std::array<size_t,2>{length, length});

        std::string outname = alps::fs::remove_extensions(arname) + ".coeffs.txt";
        std::ofstream os(outname);
        for (size_t i = 0; i < length; ++i) {
            for (size_t j = 0; j < length; ++j) {
                os << C[i][j] << '\t';
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
