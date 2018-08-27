#pragma once

#include <initializer_list>
#include <vector>


template <typename Point>
struct polygon {
    using point_type = Point;

    polygon(std::initializer_list<point_type> il) {
        points.reserve(il.size());
        for (auto const& p : il)
            points.push_back(p);
    }

    bool is_inside (point_type test) const {
        bool inside = false;
        auto p2 = points.begin();
        auto p1 = p2++;
        for (; p1 != points.end(); ++p1, ++p2) {
            if (p2 == points.end())
                p2 = points.begin();
            auto c1 = p1->begin();
            auto c2 = p2->begin();
            auto ctest = test.begin();
            if ((*ctest <= *c1 && *ctest < *c2) || (*ctest >= *c1 && *ctest > *c2))
                continue;
            double x = (*ctest - *c1) / (*c2 - *c1);
            ++c1, ++c2, ++ctest;
            double y = (*c1) * (1. - x) + (*c2) * x;
            inside ^= y < *ctest;
        }
        return inside;
    }
private:
    std::vector<point_type> points;
};
