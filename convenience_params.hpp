#pragma once

#include <alps/params.hpp>
#include "filesystem.hpp"

inline alps::params& define_convenience_parameters(alps::params & parameters) {
    const std::string origin = alps::origin_name(parameters);
    parameters
        .define<std::size_t>("timelimit", 0, "time limit for the simulation")
        .define<std::string>("outputfile",
                             replace_extension(origin, ".out.h5"),
                             "name of the output file")
        .define<std::string>("checkpoint",
                             replace_extension(origin, ".clone.h5"),
                             "name of the checkpoint file to save to")
        ;
    return parameters;
}
