#include "lastvector/collision.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lv {

namespace {

constexpr float kEpsilon = 1e-6f;

float sqr(float v) { return v * v; }

bool point_inside_aabb(Vec2 p, const Obstacle& box) {
    return p.x >= box.x && p.x <= (box.x + box.w) && p.y >= box.y && p.y <= (box.y + box.h);
}

} // namespace

float clamp(float x, float lo, float hi) {
    return std::max(lo, std::min(x, hi));
}

Vec2 closest_point_on_aabb(Vec2 point, const Obstacle& box) {
    return {
        clamp(point.x, box.x, box.x + box.w),
        clamp(point.y, box.y, box.y + box.h),
    };
}

bool circle_vs_aabb_resolve(Vec2& center, float radius, const Obstacle& box) {
    const Vec2 closest = closest_point_on_aabb(center, box);
    const float dx = center.x - closest.x;
    const float dy = center.y - closest.y;
    const float dist_sq = dx * dx + dy * dy;
    const float radius_sq = radius * radius;

    if (dist_sq < radius_sq && dist_sq > kEpsilon) {
        const float dist = std::sqrt(dist_sq);
        const float penetration = radius - dist;
        const float inv_dist = 1.0f / dist;
        center.x += dx * inv_dist * penetration;
        center.y += dy * inv_dist * penetration;
        return true;
    }

    if (point_inside_aabb(center, box) || dist_sq <= kEpsilon) {
        const float left = center.x - box.x;
        const float right = (box.x + box.w) - center.x;
        const float top = center.y - box.y;
        const float bottom = (box.y + box.h) - center.y;

        const float min_push = std::min(std::min(left, right), std::min(top, bottom));
        if (min_push == left) {
            center.x = box.x - radius;
        } else if (min_push == right) {
            center.x = box.x + box.w + radius;
        } else if (min_push == top) {
            center.y = box.y - radius;
        } else {
            center.y = box.y + box.h + radius;
        }
        return true;
    }

    return false;
}

bool circle_vs_aabb_overlap(Vec2 center, float radius, const Obstacle& box) {
    const Vec2 closest = closest_point_on_aabb(center, box);
    const float dx = center.x - closest.x;
    const float dy = center.y - closest.y;
    return (dx * dx + dy * dy) <= sqr(radius);
}

float ray_intersect_aabb(Vec2 origin, Vec2 dir, const Obstacle& box) {
    float tmin = -std::numeric_limits<float>::infinity();
    float tmax = std::numeric_limits<float>::infinity();

    const float min_x = box.x;
    const float max_x = box.x + box.w;
    const float min_y = box.y;
    const float max_y = box.y + box.h;

    if (std::abs(dir.x) < kEpsilon) {
        if (origin.x < min_x || origin.x > max_x) return std::numeric_limits<float>::infinity();
    } else {
        float tx1 = (min_x - origin.x) / dir.x;
        float tx2 = (max_x - origin.x) / dir.x;
        if (tx1 > tx2) std::swap(tx1, tx2);
        tmin = std::max(tmin, tx1);
        tmax = std::min(tmax, tx2);
    }

    if (std::abs(dir.y) < kEpsilon) {
        if (origin.y < min_y || origin.y > max_y) return std::numeric_limits<float>::infinity();
    } else {
        float ty1 = (min_y - origin.y) / dir.y;
        float ty2 = (max_y - origin.y) / dir.y;
        if (ty1 > ty2) std::swap(ty1, ty2);
        tmin = std::max(tmin, ty1);
        tmax = std::min(tmax, ty2);
    }

    if (tmax < 0.0f || tmin > tmax) return std::numeric_limits<float>::infinity();
    if (tmin >= 0.0f) return tmin;
    if (tmax >= 0.0f) return tmax;
    return std::numeric_limits<float>::infinity();
}

float ray_intersect_circle(Vec2 origin, Vec2 dir, Vec2 center, float radius) {
    const Vec2 m{origin.x - center.x, origin.y - center.y};
    const float b = m.x * dir.x + m.y * dir.y;
    const float c = m.x * m.x + m.y * m.y - radius * radius;

    if (c <= 0.0f) return 0.0f;
    const float disc = b * b - c;
    if (disc < 0.0f) return std::numeric_limits<float>::infinity();

    const float sqrt_disc = std::sqrt(disc);
    const float t0 = -b - sqrt_disc;
    if (t0 >= 0.0f) return t0;
    const float t1 = -b + sqrt_disc;
    if (t1 >= 0.0f) return t1;
    return std::numeric_limits<float>::infinity();
}

} // namespace lv
