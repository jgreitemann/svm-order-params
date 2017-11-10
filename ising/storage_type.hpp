/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <vector>
#include <alps/hdf5.hpp>

// Storage class for 2D spin array.
class storage_type {
private:
    std::vector<int> data_;
    size_t ncols;
public:
    // Constructor
    storage_type(size_t nrows, size_t ncols)
        : data_(nrows * ncols), ncols(ncols)
    {}

    // Read access
    int operator()(size_t i, size_t j) const {
        return data_[i * ncols + j];
    }

    std::vector<int> const& data () const {
        return data_;
    }

    // Read/Write access
    int& operator()(size_t i, size_t j) {
        return data_[i * ncols + j];
    }

    auto begin () { return data_.begin(); }
    auto begin () const { return data_.begin(); }
    auto end () { return data_.end(); }
    auto end () const { return data_.end(); }

    // Custom save
    void save(alps::hdf5::archive& ar) const {
        ar["data"] << data_;
        ar["ncols"] << ncols;
    }
    // Custom load
    void load(alps::hdf5::archive& ar) {
        ar["data"] >> data_;
        ar["ncols"] >> ncols;
    }
};
