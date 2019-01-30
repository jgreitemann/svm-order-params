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
#include <limits>
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

        SVM_LABEL_BEGIN(binary, 2)
        SVM_LABEL_ADD(ORDERED)
        SVM_LABEL_ADD(DISORDERED)
        SVM_LABEL_END()

        SVM_LABEL_BEGIN(D2h, 3)
        SVM_LABEL_ADD(O3)
        SVM_LABEL_ADD(Dinfh)
        SVM_LABEL_ADD(D2h)
        SVM_LABEL_END()

        SVM_LABEL_BEGIN(D3h, 3)
        SVM_LABEL_ADD(O3)
        SVM_LABEL_ADD(Dinfh)
        SVM_LABEL_ADD(D3h)
        SVM_LABEL_END()

        template <size_t nr = svm::DYNAMIC>
        struct numeric_label {
            static const size_t nr_labels = nr;
            static const size_t label_dim = 1;
            numeric_label () : val(0) {}
            template <class Iterator,
                      typename Tag = typename std::iterator_traits<Iterator>::value_type>
            numeric_label (Iterator begin) : val (floor(*begin)) {
                if (val < 0 || val >= nr_labels)
                    throw std::runtime_error(static_cast<std::stringstream&>(std::stringstream{} << "invalid label: " << val).str());
            }
            numeric_label (double x) : val (floor(x)) {
                if (val < 0. || val >= nr_labels)
                    throw std::runtime_error(static_cast<std::stringstream&>(std::stringstream{} << "invalid label: " << val).str());
            }
            operator double() const { return val; }
            double const * begin() const { return &val; }
            double const * end() const { return &val + 1; }
            friend bool operator== (numeric_label lhs, numeric_label rhs) {
                return lhs.val == rhs.val;
            }
            friend std::ostream & operator<< (std::ostream & os, numeric_label l) {
                return os << l.val;
            }
        private:
            double val;
        };

    };

    namespace classifier {

        struct critical_temperature {
            using point_type = point::temperature;
            using label_type = label::binary::label;

            static void define_parameters(alps::params & params);

            critical_temperature(alps::params const& params);
            label_type operator() (point_type pp);
        private:
            double temp_crit;
        };

        template <typename Point>
        struct orthants {
            using point_type = Point;
            using label_type = label::numeric_label<(1 << point_type::label_dim)>;

            static void define_parameters(alps::params & params) {
                point_type::define_parameters(params, "classifier.orthants.");
            }

            orthants(alps::params const& params)
                : origin(params) {}
            label_type operator() (point_type pp) {
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
        private:
            point_type origin;
        };

        template <typename Point>
        struct hyperplane {
            using point_type = Point;
            using label_type = label::binary::label;

            static void define_parameters(alps::params & params) {
                point_type::define_parameters(params, "classifier.hyperplane.support.");
                point_type::define_parameters(params, "classifier.hyperplane.normal.");
            }

            hyperplane(alps::params const& params)
                : support(params, "classifier.hyperplane.support.")
                , normal(params, "classifier.hyperplane.normal.") {}

            label_type operator() (point_type pp) {
                std::transform(pp.begin(), pp.end(), support.begin(), pp.begin(),
                               std::minus<>{});
                double res = std::inner_product(pp.begin(), pp.end(),
                                                normal.begin(), 0.);
                return res > 0 ? label::binary::ORDERED : label::binary::DISORDERED;
            }
        private:
            point_type support, normal;
        };

        template <typename Point, typename Label>
        struct phase_diagram {
            using point_type = Point;
            using label_type = Label;
            using map_type = std::map<std::string, phase_diagram>;
            using pair_type = std::pair<label_type, polygon<point_type>>;

            static void define_parameters(alps::params & params) {
                params.define<std::string>("classifier.phase_diagram.name",
                                           "key of the phase diagram map entry");
            }

            static inline auto get_map();

            phase_diagram(alps::params const& params)
                : phase_diagram([&] {
                        try {
                            return get_map().at(params["classifier.phase_diagram.name"]);
                        } catch (...) {
                            std::stringstream ss;
                            ss << "unknown phase diagram \""
                               << params["classifier.phase_diagram.name"].as<std::string>()
                               << "\"";
                            throw std::runtime_error(ss.str());
                        }
                    }()) {}

            phase_diagram(std::initializer_list<pair_type> il) {
                pairs.reserve(il.size());
                for (auto const& p : il)
                    pairs.push_back(p);
            }

            label_type operator() (point_type pp) {
                for (auto const& p : pairs) {
                    if (p.second.is_inside(pp))
                        return p.first;
                }
                throw std::runtime_error("phase diagram point not contained in "
                                         "any polygon");
                return label_type();
            }
        private:
            std::vector<pair_type> pairs;
        };

        using D2h_phase_diagram = phase_diagram<point::J1J3, label::D2h::label>;
        extern typename D2h_phase_diagram::map_type D2h_map;

        using D3h_phase_diagram = phase_diagram<point::J1J3, label::D3h::label>;
        extern typename D3h_phase_diagram::map_type D3h_map;

        template <>
        inline auto phase_diagram<point::J1J3, label::D2h::label>::get_map() {
            return D2h_map;
        }

        template <>
        inline auto phase_diagram<point::J1J3, label::D3h::label>::get_map() {
            return D3h_map;
        }

    }

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

        struct gaussian_temperatures : public policy<point::temperature> {
            static void define_parameters(alps::params & params);
            gaussian_temperatures (alps::params const& params);
            virtual size_t size() const final override;
            virtual bool yield (point_type & point, rng_type & rng) final override;
        private:
            size_t N;
            double temp_center;
            double temp_min;
            double temp_max;
            double temp_step;
            double temp_sigma_sq;
        };

        struct uniform_temperatures : public policy<point::temperature> {
            static void define_parameters(alps::params & params);
            uniform_temperatures (alps::params const& params);
            virtual size_t size() const final override;
            virtual bool yield (point_type & point, rng_type & rng) final override;
        private:
            size_t N;
            double temp_min;
            double temp_max;
            double temp_step;
        };

        struct equidistant_temperatures : public policy<point::temperature> {
            static void define_parameters(alps::params & params);
            equidistant_temperatures (alps::params const& params, size_t N, size_t n=0);
            virtual size_t size() const final override;
            virtual bool yield (point_type & point, rng_type &) final override;
            virtual void save (alps::hdf5::archive & ar) const final override;
            virtual void load (alps::hdf5::archive & ar) final override;

        private:
            size_t N, n;
            double temp_max;
            double temp_step;
            bool cooling;
        };

        template <typename Point>
        struct cycle : public policy<Point> {
            using point_type = typename policy<Point>::point_type;
            using rng_type = typename policy<Point>::rng_type;
            static const size_t MAX_CYCLE = 8;

            static void define_parameters(alps::params & params) {
                for (size_t i = 1; i <= MAX_CYCLE; ++i)
                    point_type::define_parameters(params, format_prefix(i));
            }

            cycle (std::initializer_list<point_type> il, size_t offset = 0)
                : n(offset)
            {
                for (auto p : il)
                    points.push_back(p);
                n = n % points.size();
            }

            cycle (alps::params const& params, size_t offset = 0)
                : n(offset)
            {
                for (size_t i = 1; i <= MAX_CYCLE && point_type::supplied(params, format_prefix(i)); ++i) {
                    points.emplace_back(params, format_prefix(i));
                }
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

            static std::string format_prefix(size_t i) {
                std::stringstream ss;
                ss << "sweep.cycle.P" << i << '.';
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
                                          std::string prefix = "sweep.")
            {
                point_type::define_parameters(params, prefix + "grid.a.");
                point_type::define_parameters(params, prefix + "grid.b.");
                for (size_t i = 1; i <= dim; ++i)
                    params.define<size_t>(format_subdiv(i, prefix), 1,
                                          "number of grid subdivisions");
            }

            grid (alps::params const& params, size_t offset = 0,
                  std::string prefix = "sweep.")
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
                                          std::string prefix = "sweep.")
            {
                for (size_t i = 1; i <= MAX_GRID; ++i)
                    point_type::define_parameters(params, format_stop(i, prefix));
                for (size_t i = 1; i <= dim; ++i)
                    params.define<size_t>(format_subdiv(i, prefix), 1,
                                          "number of grid subdivisions");
            }

            nonuniform_grid (alps::params const& params, size_t offset = 0,
                             std::string prefix = "sweep.")
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

            static void define_parameters(alps::params & params) {
                point_type::define_parameters(params, "sweep.uniform.a.");
                point_type::define_parameters(params, "sweep.uniform.b.");
            }

            uniform (size_t N, point_type a, point_type b)
                : N(N), a(a), b(b) {}

            uniform (alps::params const& params)
                : N(params["sweep.N"].as<size_t>())
                , a(params, "sweep.uniform.a.")
                , b(params, "sweep.uniform.b.") {}

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

            static void define_parameters(alps::params & params) {
                point_type::define_parameters(params, "sweep.uniform_line.a.");
                point_type::define_parameters(params, "sweep.uniform_line.b.");
            }

            uniform_line (size_t N, point_type a, point_type b)
                : N(N), a(a), b(b) {}

            uniform_line (alps::params const& params)
                : N(params["sweep.N"].as<size_t>())
                , a(params, "sweep.uniform_line.a.")
                , b(params, "sweep.uniform_line.b.") {}

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
                                          std::string prefix = "sweep.")
                {
                    point_type::define_parameters(params, prefix + "line_scan.a.");
                    point_type::define_parameters(params, prefix + "line_scan.b.");
                    params.define<size_t>(prefix + "line_scan.N", 8,
                                          "number of phase points on line");
                }

            line_scan (alps::params const& params, size_t offset = 0,
                       std::string prefix = "sweep.")
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
        void define_parameters (alps::params & params) {
            params.define<size_t>("sweep.N", 1, "number of phase space points");
            if (std::is_same<Point, point::temperature>::value) {
                gaussian_temperatures::define_parameters(params);
                uniform_temperatures::define_parameters(params);
                equidistant_temperatures::define_parameters(params);
            }
            cycle<Point>::define_parameters(params);
            grid<Point>::define_parameters(params);
            nonuniform_grid<Point>::define_parameters(params);
            uniform<Point>::define_parameters(params);
            uniform_line<Point>::define_parameters(params);
        }

        template <typename Point>
        auto from_parameters(alps::params const& parms, size_t seed_offset = 0)
        {
            return std::unique_ptr<policy<Point>>{[&] () -> policy<Point>* {
                std::string dist_name = parms["sweep.dist"];
                if (std::is_same<Point, phase_space::point::temperature>::value) {
                    if (dist_name == "gaussian")
                        return dynamic_cast<policy<Point>*>(
                            new gaussian_temperatures(parms));
                    if (dist_name == "uniform")
                        return dynamic_cast<policy<Point>*>(
                            new uniform_temperatures(parms));
                    if (dist_name == "bimodal")
                        return dynamic_cast<policy<Point>*>(
                            new equidistant_temperatures(parms, 2, seed_offset));
                }
                if (dist_name == "cycle")
                    return dynamic_cast<policy<Point>*>(
                        new cycle<Point> (parms, seed_offset));
                if (dist_name == "grid")
                    return dynamic_cast<policy<Point>*>(
                        new grid<Point> (parms, seed_offset));
                if (dist_name == "nonuniform_grid")
                    return dynamic_cast<policy<Point>*>(
                        new nonuniform_grid<Point>(parms, seed_offset));
                if (dist_name == "uniform")
                    return dynamic_cast<policy<Point>*>(
                        new uniform<Point> (parms));
                if (dist_name == "uniform_line")
                    return dynamic_cast<policy<Point>*>(
                        new uniform_line<Point> (parms));
                throw std::runtime_error("Invalid sweep policy \"" + dist_name + "\"");
                return nullptr;
            }()};
        }

    }

    namespace classifier {

        template <typename Point>
        struct fixed_from_cycle {
            using point_type = Point;
            using label_type = label::numeric_label<svm::DYNAMIC>;

            static void define_parameters(alps::params & params) {
            }

            fixed_from_cycle(alps::params const& params) {
                using cycs = sweep::cycle<point_type>;
                for (size_t i = 1; i <= cycs::MAX_CYCLE && point_type::supplied(params, cycs::format_prefix(i)); ++i) {
                    points.emplace_back(params, cycs::format_prefix(i));
                }
            }
            label_type operator() (point_type pp) {
                size_t i = 0;
                double d = std::numeric_limits<double>::max();
                point::distance<point_type> dist{};
                for (size_t j = 0; j < points.size(); ++j) {
                    double dd = dist(points[j], pp);
                    if (dd < d) {
                        i = j;
                        d = dd;
                    }
                }
                return i;
            }
        private:
            std::vector<point_type> points;
        };

        template <typename Point>
        struct fixed_from_grid {
            using point_type = Point;
            using grid_type = sweep::grid<point_type>;
            using label_type = label::numeric_label<svm::DYNAMIC>;

            static const size_t dim = point_type::label_dim;

            static void define_parameters(alps::params & params) {
            }

            fixed_from_grid (alps::params const& params)
                : a(params, "sweep.grid.a.")
                , b(params, "sweep.grid.b.")
            {
                for (size_t i = 1; i <= dim; ++i)
                    subdivs[i-1] = params[grid_type::format_subdiv(i, "sweep.")];
                size = std::accumulate(subdivs.begin(), subdivs.end(),
                                       1, std::multiplies<>());
            }

            label_type operator() (point_type pp) {
                std::array<long, dim> coords;
                auto ita = a.begin(), itb = b.begin(), itp = pp.begin();
                for (auto itc = coords.begin(), its = subdivs.begin();
                     itc != coords.end();
                     ++itc, ++its, ++ita, ++itb, ++itp)
                {
                    *itc = (*itp - *ita) / (*itb - *ita) * (*its - 1) + 0.5;
                }

                long tot = 0;
                for (auto itc = coords.rbegin(), its = subdivs.rbegin();
                     itc != coords.rend();
                     ++itc, ++its)
                {
                    tot = *itc + *its * tot;
                }

                if (tot < 0 || size <= tot)
                    throw std::runtime_error([&pp] {
                            std::stringstream ss;
                            ss << "phase point " << pp
                               << " exceeds limits of grid";
                            return ss.str();
                        }());
                return tot;
            }
        private:
            std::array<size_t, dim> subdivs;
            point_type a, b;
            size_t size;
        };

        template <typename Point>
        struct fixed_from_nonuniform_grid {
            using point_type = Point;
            using grid_type = sweep::nonuniform_grid<point_type>;
            using label_type = label::numeric_label<svm::DYNAMIC>;

            static const size_t dim = point_type::label_dim;

            static void define_parameters(alps::params & params) {
            }

            fixed_from_nonuniform_grid (alps::params const& params) {
                std::string prefix = "sweep.";
                for (size_t i = 1; i <= dim; ++i)
                    subdivs[i-1] = params[grid_type::format_subdiv(i, prefix)];
                size = std::accumulate(subdivs.begin(), subdivs.end(),
                                       1, std::multiplies<>());
                auto max = *std::max_element(subdivs.begin(), subdivs.end());
                for (size_t i = 1; i <= max; ++i)
                    ppoints.emplace_back(params,
                                         grid_type::format_stop(i, prefix));
            }

            label_type operator() (point_type pp) {
                std::array<long, dim> coords;
                std::vector<double> dists(ppoints.size());

                auto itc = coords.begin(), its = subdivs.begin(), itp = pp.begin();
                for (size_t i = 0; itc != coords.end(); ++itc, ++its, ++itp, ++i) {
                    for (size_t j = 0; j < *its; ++j)
                        dists[j] = std::abs(*itp - *std::next(ppoints[j].begin(), i));
                    *itc = std::min_element(dists.begin(), dists.begin() + *its)
                            - dists.begin();
                }

                long tot = 0;
                for (auto itc = coords.rbegin(), its = subdivs.rbegin();
                     itc != coords.rend();
                     ++itc, ++its)
                {
                    tot = *itc + *its * tot;
                }

                if (tot < 0 || size <= tot)
                    throw std::runtime_error([&pp] {
                            std::stringstream ss;
                            ss << "phase point " << pp
                               << " exceeds limits of grid";
                            return ss.str();
                        }());
                return tot;
            }
        private:
            std::array<size_t, dim> subdivs;
            std::vector<point_type> ppoints;
            size_t size;
        };

    }

}
