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

#pragma once
#include "label.hpp"
#include "polygon.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <alps/params.hpp>


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

        namespace numeric_label {
            template <size_t nr>
            struct label {
                static const size_t nr_labels = nr;
                static const size_t label_dim = 1;
                label () : val(0) {}
                template <class Iterator,
                          typename Tag = typename std::iterator_traits<Iterator>::value_type>
                label (Iterator begin) : val (floor(*begin)) {
                    if (val < 0 || val >= nr_labels)
                        throw std::runtime_error(static_cast<std::stringstream&>(std::stringstream{} << "invalid label: " << val).str());
                }
                label (double x) : val (floor(x)) {
                    if (val < 0. || val >= nr_labels)
                        throw std::runtime_error(static_cast<std::stringstream&>(std::stringstream{} << "invalid label: " << val).str());
                }
                operator double() const { return val; }
                double const * begin() const { return &val; }
                double const * end() const { return &val + 1; }
                friend bool operator== (label lhs, label rhs) {
                    return lhs.val == rhs.val;
                }
                friend std::ostream & operator<< (std::ostream & os, label l) {
                    return os << l.val;
                }
            private:
                double val;
            };
        }

    };

    namespace point {

        struct temperature {
            static const size_t label_dim = 1;
            using iterator = double *;
            using const_iterator = double const *;

            static void define_parameters(alps::params & params, std::string prefix="") {
                params.define<double>(prefix + "temp", 1., "temperature");
            }

            static bool supplied(alps::params const& params, std::string prefix="") {
                return params.supplied(prefix + "temp");
            }

            temperature() : temp(-1) {}
            temperature(double temp) : temp(temp) {}
            temperature(alps::params const& params, std::string prefix="")
                : temp(params[prefix + "temp"].as<double>()) {}

            template <class Iterator>
            temperature(Iterator begin) : temp(*begin) {}

            const_iterator begin() const { return &temp; }
            iterator begin() { return &temp; }
            const_iterator end() const { return &temp + 1; }
            iterator end() { return &temp + 1; }

            double temp;
        };

        struct J1J3 {
            static const size_t label_dim = 2;
            using iterator = double *;
            using const_iterator = double const *;

            static void define_parameters(alps::params & params, std::string prefix="") {
                params
                    .define<double>(prefix + "J1", 0., "J1 coupling")
                    .define<double>(prefix + "J3", 0., "J3 coupling");
            }

            static bool supplied(alps::params const& params, std::string prefix="") {
                return params.supplied(prefix + "J1")
                    && params.supplied(prefix + "J3");
            }

            J1J3(alps::params const& params, std::string prefix="")
                : J{params[prefix + "J1"].as<double>(),
                    params[prefix + "J3"].as<double>()} {}

            J1J3() : J{-1, -1} {}
            J1J3(double J1, double J3) : J{J1, J3} {}

            template <class Iterator>
            J1J3(Iterator begin) : J {*begin, *(++begin)} {}

            const_iterator begin() const { return J; }
            iterator begin() { return J; }
            const_iterator end() const { return J + 2; }
            iterator end() { return J + 2; }

            double const& J1() const { return J[0]; }
            double & J1() { return J[0]; }
            double const& J3() const { return J[1]; }
            double & J3() { return J[1]; }

            double J[2];
        };

        template <typename Point>
        bool operator== (Point const& lhs, Point const& rhs) {
            return std::equal(lhs.begin(), lhs.end(), rhs.begin());
        }

        template <typename Point, typename = typename std::enable_if_t<(Point::label_dim > 0)>>
        std::ostream& operator<< (std::ostream & os, Point const& p) {
            auto it = p.begin();
            os << '(' << *(it++);
            for (; it != p.end(); ++it)
                os << ", " << *it;
            return os << ')';
        }

        template <typename Point>
        struct distance {
            double operator() (Point const& lhs, Point const& rhs) const {
                double d = 0.;
                auto itl = lhs.begin();
                auto itr = rhs.begin();
                for (; itl != lhs.end(); ++itl, ++itr) {
                    d += pow(*itl - *itr, 2);
                }
                return sqrt(d);
            }
        };

    }

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
            using label_type = label::numeric_label::label<(1 << point_type::label_dim)>;

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

            static auto get_map();

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

        template <typename Point, typename Label>
        auto phase_diagram<Point, Label>::get_map() {
            return D2h_map;
        }

    }

    namespace sweep {

        template <typename Point, typename RNG = std::mt19937>
        struct policy {
            using point_type = Point;
            using rng_type = RNG;

            virtual bool yield (point_type & point, rng_type & rng) = 0;
        };

        struct gaussian_temperatures : public policy<point::temperature> {
            static void define_parameters(alps::params & params);
            gaussian_temperatures (alps::params const& params);
            virtual bool yield (point_type & point, rng_type & rng) final override;
        private:
            double temp_center;
            double temp_min;
            double temp_max;
            double temp_step;
            double temp_sigma_sq;
        };

        struct uniform_temperatures : public policy<point::temperature> {
            static void define_parameters(alps::params & params);
            uniform_temperatures (alps::params const& params);
            virtual bool yield (point_type & point, rng_type & rng) final override;
        private:
            double temp_min;
            double temp_max;
            double temp_step;
        };

        struct equidistant_temperatures : public policy<point::temperature> {
            static void define_parameters(alps::params & params);
            equidistant_temperatures (alps::params const& params, size_t N, size_t n=0);
            virtual bool yield (point_type & point, rng_type &) final override;
        private:
            size_t n, N;
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

            virtual bool yield (point_type & point, rng_type &) final override {
                point = points[n];
                n = (n + 1) % points.size();
                return true;
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
        struct uniform_line : public policy<Point> {
            using point_type = typename policy<Point>::point_type;
            using rng_type = typename policy<Point>::rng_type;

            static void define_parameters(alps::params & params) {
                point_type::define_parameters(params, "sweep.uniform_line.a.");
                point_type::define_parameters(params, "sweep.uniform_line.b.");
            }

            uniform_line (point_type a, point_type b)
                : a(a), b(b) {}

            uniform_line (alps::params const& params)
                : a(params, "sweep.uniform_line.a.")
                , b(params, "sweep.uniform_line.b.") {}

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
            point_type a, b;
        };

        template <typename Point>
        struct line_scan : public policy<Point> {
            using point_type = typename policy<Point>::point_type;
            using rng_type = typename policy<Point>::rng_type;

            line_scan (point_type const& a, point_type const& b, size_t N)
                : a(a), b(b), n(0), N(N) {}

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
        private:
            const point_type a, b;
            size_t n, N;
        };

        template <typename Point>
        void define_parameters (alps::params & params) {
            if (std::is_same<Point, point::temperature>::value) {
                gaussian_temperatures::define_parameters(params);
                uniform_temperatures::define_parameters(params);
                equidistant_temperatures::define_parameters(params);
            }
            cycle<Point>::define_parameters(params);
            uniform_line<Point>::define_parameters(params);
        }

    }

    namespace classifier {

        template <typename Point, size_t M>
        struct fixed_from_cycle {
            using point_type = Point;
            using label_type = label::numeric_label::label<M>;

            static void define_parameters(alps::params & params) {
            }

            fixed_from_cycle(alps::params const& params) {
                for (size_t i = 1; i <= M; ++i) {
                    points.emplace_back(params, sweep::cycle<point_type>::format_prefix(i));
                }
            }
            label_type operator() (point_type pp) {
                auto it = pp.begin();
                if (*it > 5 || *(++it) > 5)
                    std::cout << "buh!" << std::endl;
                size_t i;
                double d = std::numeric_limits<double>::max();
                point::distance<point_type> dist{};
                for (size_t j = 0; j < M; ++j) {
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

    }

}
