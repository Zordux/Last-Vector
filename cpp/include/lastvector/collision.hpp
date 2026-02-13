#pragma once

#include "state.hpp"

namespace lv {

Vec2 closest_point_on_aabb(Vec2 point, const Obstacle& box);
bool circle_vs_aabb_resolve(Vec2& center, float radius, const Obstacle& box);
bool circle_vs_aabb_overlap(Vec2 center, float radius, const Obstacle& box);

float ray_intersect_aabb(Vec2 origin, Vec2 dir, const Obstacle& box);
float ray_intersect_circle(Vec2 origin, Vec2 dir, Vec2 center, float radius);

} // namespace lv
