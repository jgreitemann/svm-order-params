#pragma once

#include <string>

#include <alps/params.hpp>

#include "argh.h"


template <typename T>
bool override_parameter (std::string const& name, alps::params & parameters, argh::parser & cmdl) {
    T new_param;
    if (cmdl(name) >> new_param) {
        std::cout << "override parameter " << name << ": " << new_param << std::endl;
        parameters[name] = new_param;
        return true;
    }
    return false;
}
