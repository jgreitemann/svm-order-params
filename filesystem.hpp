#pragma once

#include <experimental/filesystem>


namespace fs = std::experimental::filesystem;


inline std::string replace_extension (std::string const& ini_or_h5, std::string const& new_ext) {
    fs::path ext;
    fs::path known_exts[] = {
        ".ini",
        ".h5",
        ".txt",
        ".ppm",
        ".out",
        ".clone",
        ".test"
    };
    fs::path p(ini_or_h5);
    for (; !(ext = p.extension()).empty(); p = p.stem()) {
        auto it = std::find(std::begin(known_exts), std::end(known_exts), ext);
        if (it == std::end(known_exts))
            break;
    }
    return p += new_ext;
}
