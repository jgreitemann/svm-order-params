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

#pragma once
#include "label.hpp"
#include "phase_space_point.hpp"
#include "polygon.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <alps/params.hpp>
#include <alps/hdf5.hpp>


namespace phase_space {

    namespace label {

        template <size_t nr = svm::DYNAMIC>
        struct numeric_label {
            static const size_t nr_labels = nr;
            static const size_t label_dim = 1;
            numeric_label () : val(0) {}
            template <class Iterator,
                      typename Tag = typename std::iterator_traits<Iterator>::value_type>
            numeric_label(Iterator begin) : val (floor(*begin)) {
                if (val < 0 || val >= nr_labels)
                    throw std::runtime_error(static_cast<std::stringstream&>(
                        std::stringstream{} << "invalid label: " << val).str());
            }
            numeric_label(double x) : val (floor(x)) {
                if (val < 0. || val >= nr_labels)
                    throw std::runtime_error(static_cast<std::stringstream&>(
                        std::stringstream{} << "invalid label: " << val).str());
            }
            operator double() const { return val; }
            explicit operator size_t() const {
                return static_cast<size_t>(val + 0.5);
            }
            double const * begin() const { return &val; }
            double const * end() const { return &val + 1; }
            friend bool operator== (numeric_label lhs, numeric_label rhs) {
                return lhs.val == rhs.val;
            }
            friend std::ostream & operator<<(std::ostream & os, numeric_label l) {
                return os << l.val;
            }
        private:
            double val;
        };

    };

    namespace sweep {

        template <typename Point, typename RNG = std::mt19937>
        struct policy {
            using point_type = Point;
            using rng_type = RNG;

            virtual size_t size() const = 0;
            virtual bool yield (point_type & point, rng_type & rng) = 0;
            virtual void save (alps::hdf5::archive & ar) const {}
            virtual void load (alps::hdf5::archive & ar) {}
        };

        template <typename Point>
        struct cycle : public policy<Point> {
            using point_type = typename policy<Point>::point_type;
            using rng_type = typename policy<Point>::rng_type;
            static const size_t MAX_CYCLE = 8;

            static void define_parameters(alps::params & params,
                std::string const& prefix)
            {
                for (size_t i = 1; i <= MAX_CYCLE; ++i)
                    point_type::define_parameters(params,
                        format_prefix(i, prefix));
            }

            cycle (std::initializer_list<point_type> il, size_t offset = 0)
                : n(offset)
            {
                for (auto p : il)
                    points.push_back(p);
                if (points.empty())
                    throw std::runtime_error("cycle sweep policy "
                        "list-initialized but no points supplied");
                n = n % points.size();
            }

            cycle (alps::params const& params,
                   std::string const& prefix,
                   size_t offset = 0)
                : n(offset)
            {
                for (size_t i = 1; i <= MAX_CYCLE
                    && point_type::supplied(params, format_prefix(i, prefix)); ++i)
                {
                    points.emplace_back(params, format_prefix(i, prefix));
                }
                if (points.empty())
                    throw std::runtime_error("cycle sweep policy initialized "
                        "but no points supplied in parameters");
                n = n % points.size();
            }

            virtual size_t size() const final override {
                return points.size();
            }

            virtual bool yield (point_type & point, rng_type &) final override {
                point = points[n];
                n = (n + 1) % points.size();
                return true;
            }

            virtual void save (alps::hdf5::archive & ar) const final override {
                ar["n"] << n;
            }

            virtual void load (alps::hdf5::archive & ar) final override {
                ar["n"] >> n;
            }

            static std::string format_prefix(size_t i,
                std::string const& prefix)
            {
                std::stringstream ss;
                ss << prefix << "cycle.P" << i << '.';
                return ss.str();
            }

        private:
            std::vector<point_type> points;
            size_t n;
        };

        template <typename Point>
        struct grid : public policy<Point> {
            using point_type = typename policy<Point>::point_type;
            using rng_type = typename policy<Point>::rng_type;

            static const size_t dim = point_type::label_dim;

            static void define_parameters(alps::params & params,
                                          std::string const& prefix)
            {
                point_type::define_parameters(params, prefix + "grid.a.");
                point_type::define_parameters(params, prefix + "grid.b.");
                for (size_t i = 1; i <= dim; ++i)
                    params.define<size_t>(format_subdiv(i, prefix), 1,
                                          "number of grid subdivisions");
            }

            grid(alps::params const& params,
                 std::string const& prefix,
                 size_t offset = 0)
                : a(params, prefix + "grid.a.")
                , b(params, prefix + "grid.b.")
                , n(offset)
            {
                for (size_t i = 1; i <= dim; ++i)
                    subdivs[i-1] = params[format_subdiv(i, prefix)].as<size_t>();

                double frac = 0.;
                for (double p = 0.5; offset != 0; offset >>= 1, p /= 2)
                    if (offset & 1)
                        frac += p;
                n = size() * frac;
            }

            virtual size_t size() const final override {
                return std::accumulate(subdivs.begin(), subdivs.end(),
                                       1, std::multiplies<>{});
            }

            bool yield (point_type & point) {
                size_t x = n;
                auto it = point.begin();
                auto ita = a.begin();
                auto itb = b.begin();
                for (size_t i = 0; i < dim; ++i, ++it, ++ita, ++itb) {
                    if (subdivs[i] == 1) {
                        *it = *ita;
                    } else {
                        *it = *ita + (*itb - *ita) / (subdivs[i] - 1) * (x % subdivs[i]);
                        x /= subdivs[i];
                    }
                }
                n = (n + 1) % size();
                return true;
            }

            virtual bool yield (point_type & point, rng_type &) final override {
                return yield(point);
            }

            virtual void save (alps::hdf5::archive & ar) const final override {
                ar["n"] << n;
            }

            virtual void load (alps::hdf5::archive & ar) final override {
                ar["n"] >> n;
            }

            static std::string format_subdiv(size_t i,
                                             std::string const& prefix)
            {
                std::stringstream ss;
                ss << prefix + "grid.N" << i;
                return ss.str();
            }

        private:
            std::array<size_t, dim> subdivs;
            point_type a, b;
            size_t n;
        };

        template <typename Point>
        struct nonuniform_grid : public policy<Point> {
            using point_type = typename policy<Point>::point_type;
            using rng_type = typename policy<Point>::rng_type;

            static const size_t dim = point_type::label_dim;
            static const size_t MAX_GRID = 10;

            static void define_parameters(alps::params & params,
                                          std::string const& prefix)
            {
                for (size_t i = 1; i <= MAX_GRID; ++i)
                    point_type::define_parameters(params, format_stop(i, prefix));
                for (size_t i = 1; i <= dim; ++i)
                    params.define<size_t>(format_subdiv(i, prefix), 1,
                                          "number of grid subdivisions");
            }

            nonuniform_grid (alps::params const& params,
                             std::string const& prefix,
                             size_t offset = 0)
                : n(offset)
            {
                for (size_t i = 1; i <= dim; ++i)
                    subdivs[i-1] = params[format_subdiv(i, prefix)].as<size_t>();
                auto max = *std::max_element(subdivs.begin(), subdivs.end());
                for (size_t i = 1; i <= max; ++i)
                    ppoints.emplace_back(params, format_stop(i, prefix));

                double frac = 0.;
                for (double p = 0.5; offset != 0; offset >>= 1, p /= 2)
                    if (offset & 1)
                        frac += p;
                n = size() * frac;
            }

            virtual size_t size() const final override {
                return std::accumulate(subdivs.begin(), subdivs.end(),
                                       1, std::multiplies<>{});
            }

            bool yield (point_type & point) {
                size_t x = n;
                auto it = point.begin();
                for (size_t i = 0; i < dim; ++i, ++it) {
                    *it = *std::next(ppoints[x % subdivs[i]].begin(), i);
                    x /= subdivs[i];
                }
                n = (n + 1) % size();
                return true;
            }

            virtual bool yield (point_type & point, rng_type &) final override {
                return yield(point);
            }

            virtual void save (alps::hdf5::archive & ar) const final override {
                ar["n"] << n;
            }

            virtual void load (alps::hdf5::archive & ar) final override {
                ar["n"] >> n;
            }

            static std::string format_subdiv(size_t i,
                                             std::string const& prefix)
            {
                std::stringstream ss;
                ss << prefix + "nonuniform_grid.N" << i;
                return ss.str();
            }

            static std::string format_stop(size_t i,
                                           std::string const& prefix)
            {
                std::stringstream ss;
                ss << prefix + "nonuniform_grid.stop" << i << '.';
                return ss.str();
            }

        private:
            std::array<size_t, dim> subdivs;
            std::vector<point_type> ppoints;
            size_t n;
        };

        template <typename Point>
        struct uniform : public policy<Point> {
            using point_type = typename policy<Point>::point_type;
            using rng_type = typename policy<Point>::rng_type;

            static void define_parameters(alps::params & params,
                                          std::string const& prefix)
            {
                point_type::define_parameters(params, prefix + "uniform.a.");
                point_type::define_parameters(params, prefix + "uniform.b.");
                params.define<size_t>(prefix + "uniform.N", 10,
                    "number of uniform point to draw");
            }

            uniform (size_t N, point_type a, point_type b)
                : N(N), a(a), b(b) {}

            uniform (alps::params const& params, std::string const& prefix)
                : N(params[prefix + "uniform.N"].as<size_t>())
                , a(params, prefix + "uniform.a.")
                , b(params, prefix + "uniform.b.") {}

            virtual size_t size() const final override {
                return N;
            }

            virtual bool yield (point_type & point, rng_type & rng) final override {
                auto ita = a.begin();
                auto itb = b.begin();
                for (auto & c : point) {
                    c = std::uniform_real_distribution<double>{*ita, *itb}(rng);
                    ++ita, ++itb;
                }
                return true;
            }

        private:
            size_t N;
            point_type a, b;
        };

        template <typename Point>
        struct uniform_line : public policy<Point> {
            using point_type = typename policy<Point>::point_type;
            using rng_type = typename policy<Point>::rng_type;

            static void define_parameters(alps::params & params,
                                          std::string const& prefix)
            {
                point_type::define_parameters(params, prefix + "uniform_line.a.");
                point_type::define_parameters(params, prefix + "uniform_line.b.");
                params.define<size_t>(prefix + "uniform_line.N", 10,
                    "number of uniform point to draw");
            }

            uniform_line (size_t N, point_type a, point_type b)
                : N(N), a(a), b(b) {}

            uniform_line (alps::params const& params, std::string const& prefix)
                : N(params[prefix + "uniform_line.N"].as<size_t>())
                , a(params, prefix + "uniform_line.a.")
                , b(params, prefix + "uniform_line.b.") {}

            virtual size_t size() const final override {
                return N;
            }

            virtual bool yield (point_type & point, rng_type & rng) final override {
                double x = std::uniform_real_distribution<double>{}(rng);
                auto ita = a.begin();
                auto itb = b.begin();
                for (auto & c : point) {
                    c = (1. - x) * (*ita) + x * (*itb);
                    ++ita, ++itb;
                }
                return true;
            }

        private:
            size_t N;
            point_type a, b;
        };

        template <typename Point>
        struct line_scan : public policy<Point> {
            using point_type = typename policy<Point>::point_type;
            using rng_type = typename policy<Point>::rng_type;

            static void define_parameters(alps::params & params,
                                          std::string const& prefix)
                {
                    point_type::define_parameters(params, prefix + "line_scan.a.");
                    point_type::define_parameters(params, prefix + "line_scan.b.");
                    params.define<size_t>(prefix + "line_scan.N", 8,
                                          "number of phase points on line");
                }

            line_scan (alps::params const& params,
                       std::string const& prefix,
                       size_t offset = 0)
                : a(params, prefix + "line_scan.a.")
                , b(params, prefix + "line_scan.b.")
                , n(offset)
                , N(params[prefix + "line_scan.N"].as<size_t>())
            {
            }

            line_scan (point_type const& a, point_type const& b, size_t N)
                : a(a), b(b), n(0), N(N) {}

            virtual size_t size() const final override {
                return N;
            }

            virtual bool yield (point_type & point, rng_type &) final override {
                return yield(point);
            }

            bool yield (point_type & point) {
                if (n == N)
                    return false;
                auto it_a = a.begin();
                auto it_b = b.begin();
                double x = 1. * n / (N - 1);
                for (auto & c : point) {
                    c = *it_b * x + *it_a * (1. - x);
                    ++it_a, ++it_b;
                }
                ++n;
                return true;
            }

            virtual void save (alps::hdf5::archive & ar) const final override {
                ar["n"] << n;
            }

            virtual void load (alps::hdf5::archive & ar) final override {
                ar["n"] >> n;
            }

        private:
            const point_type a, b;
            size_t n, N;
        };

        template <typename Point>
        struct log_scan : public policy<Point> {
            using point_type = Point;
            using rng_type = typename policy<Point>::rng_type;

            static void define_parameters(alps::params & params,
                                          std::string const& prefix)
            {
                point_type::define_parameters(params, prefix + "log_scan.a.");
                point_type::define_parameters(params, prefix + "log_scan.b.");
                for (size_t i = 0; i < point_type::label_dim; ++i) {
                    params.define<bool>(
                        prefix + "log_scan.log" + std::to_string(i + 1), true,
                        "boolean indicating if axis" + std::to_string(i + 1)
                        + " is logarithmic");
                }
                params.define<size_t>(prefix + "log_scan.N", 8,
                    "number of subdivisions");
            }

            log_scan (alps::params const& params,
                      std::string const& prefix,
                      size_t offset = 0)
                : a(params, prefix + "log_scan.a.")
                , b(params, prefix + "log_scan.b.")
                , N(params[prefix + "log_scan.N"].as<size_t>())
                , n(offset)
            {
                for (size_t i = 0; i < point_type::label_dim; ++i) {
                    is_log[i] = params[prefix + "log_scan.log"
                        + std::to_string(i + 1)].as<bool>();
                }
            }

            size_t size() const {
                return N;
            }

            virtual bool yield (point_type & point, rng_type &) final override {
                return yield(point);
            }

            bool yield (point_type & point) {
                auto it_a = a.begin();
                auto it_b = b.begin();
                auto it = point.begin();
                for (size_t i = 0; i < point_type::label_dim;
                     ++i, ++it_a, ++it_b, ++it)
                {
                    if (is_log[i])
                        *it = *it_a * pow(*it_b / *it_a, 1. * n / (N - 1));
                    else
                        *it = *it_a + (*it_b - *it_a) * n / (N - 1);
                }
                ++n;
                return true;
            }

            virtual void save (alps::hdf5::archive & ar) const final override {
                ar["n"] << n;
            }

            virtual void load (alps::hdf5::archive & ar) final override {
                ar["n"] >> n;
            }

        private:
            const point_type a, b;
            std::array<bool, point_type::label_dim> is_log;
            size_t N;
            size_t n;
        };

        template <typename Point>
        void define_parameters(alps::params & params,
                               std::string const& prefix)
        {
            cycle<Point>::define_parameters(params, prefix);
            grid<Point>::define_parameters(params, prefix);
            nonuniform_grid<Point>::define_parameters(params, prefix);
            uniform<Point>::define_parameters(params, prefix);
            uniform_line<Point>::define_parameters(params, prefix);
            line_scan<Point>::define_parameters(params, prefix);
            log_scan<Point>::define_parameters(params, prefix);
        }

        template <typename Point>
        auto from_parameters(alps::params const& params,
                             std::string const& prefix,
                             size_t seed_offset = 0)
        {
            return std::unique_ptr<policy<Point>>{[&] () -> policy<Point>* {
                std::string pol_name = params[prefix + "policy"];
                if (pol_name == "cycle")
                    return dynamic_cast<policy<Point>*>(
                        new cycle<Point>(params, prefix, seed_offset));
                if (pol_name == "grid")
                    return dynamic_cast<policy<Point>*>(
                        new grid<Point>(params, prefix, seed_offset));
                if (pol_name == "nonuniform_grid")
                    return dynamic_cast<policy<Point>*>(
                        new nonuniform_grid<Point>(params, prefix, seed_offset));
                if (pol_name == "uniform")
                    return dynamic_cast<policy<Point>*>(
                        new uniform<Point>(params, prefix));
                if (pol_name == "uniform_line")
                    return dynamic_cast<policy<Point>*>(
                        new uniform_line<Point>(params, prefix));
                if (pol_name == "line_scan")
                    return dynamic_cast<policy<Point>*>(
                        new line_scan<Point> (params, prefix, seed_offset));
                if (pol_name == "log_scan")
                    return dynamic_cast<policy<Point>*>(
                        new log_scan<Point> (params, prefix, seed_offset));
                throw std::runtime_error("Invalid sweep policy \""
                    + pol_name + "\"");
                return nullptr;
            }()};
        }

    }

    namespace classifier {

        template <typename Point>
        struct policy {
            using point_type = Point;
            using label_type = label::numeric_label<>;

            virtual label_type operator()(point_type) = 0;
            virtual std::string name(label_type const& l) const = 0;
            virtual size_t size() const = 0;

            auto get_functor() {
                return [this](point_type pp) {
                    return (*this)(pp);
                };
            }
        };

        struct critical_temperature : policy<point::temperature> {
            static void define_parameters(alps::params &, std::string const&);

            critical_temperature(alps::params const&, std::string const&);
            virtual label_type operator() (point_type pp) override;
            virtual std::string name(label_type const& l) const override;
            virtual size_t size() const override;
        private:
            static const std::string names[];
            double temp_crit;
        };

        template <typename Point>
        struct orthants : policy<Point> {
            using typename policy<Point>::point_type;
            using typename policy<Point>::label_type;

            static void define_parameters(alps::params & params,
                                          std::string const& prefix)
            {
                point_type::define_parameters(params, prefix + "orthants.");
            }

            orthants(alps::params const& params,
                     std::string const& prefix)
                : origin(params, prefix + "orthants.") {}
            virtual label_type operator()(point_type pp) override {
                size_t res = 0;
                auto oit = origin.begin();
                for (double x : pp) {
                    res *= 2;
                    if (x > *oit)
                        res += 1;
                    ++oit;
                }
                return label_type {double(res)};
            }
            virtual std::string name(label_type const& l) const override {
                std::stringstream ss;
                ss << l;
                return ss.str();
            }
            virtual size_t size() const override {
                return 1 << point_type::label_dim;
            }
        private:
            point_type origin;
        };

        template <typename Point>
        struct hyperplane : policy<Point> {
            using typename policy<Point>::point_type;
            using typename policy<Point>::label_type;

            static void define_parameters(alps::params & params,
                                          std::string const& prefix)
            {
                point_type::define_parameters(params, prefix + "hyperplane.support.");
                point_type::define_parameters(params, prefix + "hyperplane.normal.");
            }

            hyperplane(alps::params const& params,
                       std::string const& prefix)
                : support(params, prefix + "hyperplane.support.")
                , normal(params, prefix + "hyperplane.normal.") {}

            virtual label_type operator()(point_type pp) override {
                std::transform(pp.begin(), pp.end(), support.begin(), pp.begin(),
                               std::minus<>{});
                double res = std::inner_product(pp.begin(), pp.end(),
                                                normal.begin(), 0.);
                return res > 0 ? label_type{1.} : label_type{0.};
            }

            virtual std::string name(label_type const& l) const override {
                return names[size_t(l)];
            }

            virtual size_t size() const override {
                return 2;
            }

        private:
            static const std::string names[];
            point_type support, normal;
        };

        template <typename Point>
        const std::string hyperplane<Point>::names[] = {
            "DISORDERED",
            "ORDERED",
        };

        template <typename Point>
        struct phase_diagram;

        template <typename Point>
        struct phase_diagram_database {
            static const typename phase_diagram<Point>::map_type map;
        };

        template <typename Point>
        const typename phase_diagram<Point>::map_type
        phase_diagram_database<Point>::map {};

        template <typename Point>
        struct phase_diagram : policy<Point> {
            using typename policy<Point>::point_type;
            using typename policy<Point>::label_type;
            using database_type = phase_diagram_database<point_type>;
            using map_type = std::map<std::string, phase_diagram>;
            using pair_type = std::pair<std::string, polygon<point_type>>;

            static void define_parameters(alps::params & params,
                                          std::string const& prefix)
            {
                params.define<std::string>("classifier.phase_diagram.name",
                                           "key of the phase diagram map entry");
            }

            phase_diagram(alps::params const& params,
                          std::string const& prefix)
                : phase_diagram([&] {
                        try {
                            return database_type::map.at(
                                params[prefix + "phase_diagram.name"]);
                        } catch (...) {
                            std::stringstream ss;
                            ss << "unknown phase diagram \""
                               << params[prefix + "phase_diagram.name"].as<std::string>()
                               << "\"";
                            throw std::runtime_error(ss.str());
                        }
                    }()) {}

            phase_diagram(std::initializer_list<pair_type> il) {
                pairs.reserve(il.size());
                for (auto const& p : il)
                    pairs.push_back(p);
            }

            virtual label_type operator()(point_type pp) override {
                double l = 0.;
                for (auto const& p : pairs) {
                    if (p.second.is_inside(pp))
                        return {l};
                    l += 1;
                }
                throw std::runtime_error("phase diagram point not contained in "
                                         "any polygon");
                return label_type();
            }

            virtual std::string name(label_type const& l) const override {
                return pairs[size_t(l)].first;
            }

            virtual size_t size() const override {
                return pairs.size();
            }
        private:
            std::vector<pair_type> pairs;
        };

        template <>
        struct phase_diagram_database<point::J1J3> {
            static const typename phase_diagram<point::J1J3>::map_type map;
        };

        template <typename Point>
        struct fixed_from_sweep : policy<Point> {
            using typename policy<Point>::point_type;
            using typename policy<Point>::label_type;

            static void define_parameters(alps::params &, std::string const&) {
            }

            fixed_from_sweep(alps::params const& parameters,
                             std::string const&)
            {
                auto sweep_pol = phase_space::sweep::from_parameters<point_type>(
                    parameters, "sweep.");
                std::mt19937 rng{parameters["SEED"].as<size_t>()};
                std::generate_n(std::back_inserter(points), sweep_pol->size(),
                    [&, p=point_type{}]() mutable {
                        sweep_pol->yield(p, rng);
                        return p;
                    });
            }

            virtual label_type operator()(point_type pp) override {
                if (pp == infty)
                    return {static_cast<double>(size())};
                point::distance<point_type> dist{};
                auto closest_it = std::min_element(points.begin(), points.end(),
                    [&](point_type const& lhs, point_type const& rhs) {
                        return dist(lhs, pp) < dist(rhs, pp);
                    });
                return {static_cast<double>(closest_it - points.begin())};
            }

            virtual std::string name(label_type const& l) const override {
                std::stringstream ss;
                ss << 'P' << (size_t(l) + 1);
                return ss.str();
            }

            virtual size_t size() const override {
                return points.size();
            }
        private:
            std::vector<point_type> points;
            const point_type infty = point::infinity<point_type>{}();
        };

        template <typename Point>
        void define_parameters(alps::params & params,
                               std::string && prefix = "classifier.")
        {
            params.define<std::string>(prefix + "policy",
                "fixed_from_sweep",
                "name of the classifier policy");
            orthants<Point>::define_parameters(params, prefix);
            hyperplane<Point>::define_parameters(params, prefix);
            phase_diagram<Point>::define_parameters(params, prefix);
            fixed_from_sweep<Point>::define_parameters(params, prefix);
            if (std::is_same<Point, point::temperature>::value)
                critical_temperature::define_parameters(params, prefix);
        }

        template <typename Point>
        auto from_parameters(alps::params const& params,
                             std::string && prefix = "classifier.")
        {
            return std::unique_ptr<policy<Point>>{[&] () -> policy<Point>* {
                std::string pol_name = params[prefix + "policy"];
                if (pol_name == "orthants")
                    return dynamic_cast<policy<Point>*>(
                        new orthants<Point>(params, prefix));
                if (pol_name == "hyperplane")
                    return dynamic_cast<policy<Point>*>(
                        new hyperplane<Point>(params, prefix));
                if (pol_name == "phase_diagram")
                    return dynamic_cast<policy<Point>*>(
                        new phase_diagram<Point>(params, prefix));
                if (pol_name == "fixed_from_sweep")
                    return dynamic_cast<policy<Point>*>(
                        new fixed_from_sweep<Point>(params, prefix));
                if (pol_name == "critical_temperature") {
                    if (std::is_same<Point, point::temperature>::value)
                        return dynamic_cast<policy<Point>*>(
                            new critical_temperature(params, prefix));
                    else
                        throw std::runtime_error(
                            "critical_temperature classifier is only available "
                            "for use with `temperature` phase space points");
                }
                throw std::runtime_error("Invalid classifier policy \""
                    + pol_name + "\"");
                return nullptr;
            }()};
        }

    }

}
