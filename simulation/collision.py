"""Collision Detection Utilities for Q-Nav.

Provides exact 2D geometric collision and clearance calculations for:
- Circle-to-circle (drone vs cylindrical obstacles)
- Circle-to-box / AABB (drone vs rectangular structures)
- Continuous trajectory swept-circle collision (prevents tunneling through obstacles)
- Boundary containment and proximity checks
"""

from typing import Tuple, List, Optional, Any
import numpy as np


def distance_point_to_point(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def check_circle_circle(
    center1: np.ndarray,
    radius1: float,
    center2: np.ndarray,
    radius2: float,
) -> bool:
    """Check if two circles intersect or overlap.

    Args:
        center1: 2D coordinates [x, y] of first circle.
        radius1: Radius of first circle.
        center2: 2D coordinates [x, y] of second circle.
        radius2: Radius of second circle.

    Returns:
        True if the circles intersect/collide.
    """
    dist = distance_point_to_point(center1, center2)
    return dist <= (radius1 + radius2)


def distance_circle_circle(
    center1: np.ndarray,
    radius1: float,
    center2: np.ndarray,
    radius2: float,
) -> float:
    """Calculate the surface-to-surface clearance between two circles.

    Returns:
        Positive distance if separated, <= 0 if penetrating.
    """
    center_dist = distance_point_to_point(center1, center2)
    return float(center_dist - (radius1 + radius2))


def check_circle_box(
    center: np.ndarray,
    radius: float,
    box_bounds: Tuple[float, float, float, float],
) -> bool:
    """Check collision between a circle and an Axis-Aligned Bounding Box (AABB).

    Args:
        center: 2D coordinates [x, y] of circle.
        radius: Radius of circle.
        box_bounds: Tuple (x_min, y_min, x_max, y_max).

    Returns:
        True if circle intersects the box.
    """
    cx, cy = float(center[0]), float(center[1])
    x_min, y_min, x_max, y_max = box_bounds

    # Find closest point on box to circle center
    closest_x = max(x_min, min(cx, x_max))
    closest_y = max(y_min, min(cy, y_max))

    dx = cx - closest_x
    dy = cy - closest_y
    return (dx * dx + dy * dy) <= (radius * radius)


def distance_circle_box(
    center: np.ndarray,
    radius: float,
    box_bounds: Tuple[float, float, float, float],
) -> float:
    """Calculate the clearance between a circle and an AABB."""
    cx, cy = float(center[0]), float(center[1])
    x_min, y_min, x_max, y_max = box_bounds

    closest_x = max(x_min, min(cx, x_max))
    closest_y = max(y_min, min(cy, y_max))

    dx = cx - closest_x
    dy = cy - closest_y
    dist_to_box = float(np.sqrt(dx * dx + dy * dy))
    return dist_to_box - radius


def check_swept_circle_line(
    p_start: np.ndarray,
    p_end: np.ndarray,
    circle_center: np.ndarray,
    radius: float,
) -> bool:
    """Check if the capsule formed by moving a circle from p_start to p_end intersects circle_center.

    Prevents tunneling through obstacles when drone moves at high speed.
    """
    p0 = np.asarray(p_start, dtype=float)
    p1 = np.asarray(p_end, dtype=float)
    c = np.asarray(circle_center, dtype=float)

    v = p1 - p0
    seg_len_sq = float(np.dot(v, v))

    if seg_len_sq < 1e-8:
        return distance_point_to_point(p0, c) <= radius

    # Project circle center onto segment: t in [0, 1]
    t = max(0.0, min(1.0, float(np.dot(c - p0, v) / seg_len_sq)))
    projection = p0 + t * v
    dist = distance_point_to_point(projection, c)
    return dist <= radius


def check_boundary_violation(
    position: np.ndarray,
    radius: float,
    map_width: float,
    map_height: float,
) -> bool:
    """Check if a circle with given radius is out of map boundaries [0, width] x [0, height]."""
    x, y = float(position[0]), float(position[1])
    if (x - radius) < 0.0 or (x + radius) > map_width:
        return True
    if (y - radius) < 0.0 or (y + radius) > map_height:
        return True
    return False


def distance_to_boundary(
    position: np.ndarray,
    map_width: float,
    map_height: float,
) -> float:
    """Compute the minimum distance from position to any of the 4 map boundaries."""
    x, y = float(position[0]), float(position[1])
    dist_left = x
    dist_right = map_width - x
    dist_bottom = y
    dist_top = map_height - y
    return float(max(0.0, min(dist_left, dist_right, dist_bottom, dist_top)))


def predict_collision(
    p_uav_start: Optional[Any] = None,
    p_uav_end: Optional[Any] = None,
    r_uav: Optional[float] = None,
    p_obs_start: Optional[Any] = None,
    p_obs_end: Optional[Any] = None,
    r_obs: Optional[float] = None,
    *,
    uav_pos: Optional[Any] = None,
    uav_next_pos: Optional[Any] = None,
    uav_radius: Optional[float] = None,
    obs_pos: Optional[Any] = None,
    obs_next_pos: Optional[Any] = None,
    obs_velocity: Optional[Any] = None,
    obs_radius: Optional[float] = None,
    dt: float = 1.0,
    safety_margin: float = 0.0,
    **kwargs,
) -> Tuple[bool, float, float]:
    """Check continuous swept-circle collision between a moving UAV and moving obstacle.

    Parameterizes continuous movement for both bodies over normalized time tau in [0, 1]:
        p_uav(tau) = p_uav_start + tau * (p_uav_end - p_uav_start)
        p_obs(tau) = p_obs_start + tau * (p_obs_end - p_obs_start)

    Relative displacement:
        r(tau) = p_uav(tau) - p_obs(tau) = r0 + tau * v_rel
    where r0 = p_uav_start - p_obs_start and v_rel = (p_uav_end - p_uav_start) - (p_obs_end - p_obs_start).

    Solves for the exact time tau* in [0, 1] that minimizes ||r(tau)||^2.

    Returns:
        Tuple of (has_collided, min_distance, tau_min):
            - has_collided: True if minimum distance <= (r_uav + r_obs + safety_margin).
            - min_distance: Minimum center-to-center Euclidean distance achieved during step.
            - tau_min: Normalized time tau in [0, 1] at which minimum distance occurs.
    """
    # Resolve aliases
    p_u0_raw = p_uav_start if p_uav_start is not None else uav_pos
    p_u1_raw = p_uav_end if p_uav_end is not None else uav_next_pos
    r_u_raw = r_uav if r_uav is not None else (uav_radius if uav_radius is not None else 1.0)

    p_o0_raw = p_obs_start if p_obs_start is not None else obs_pos
    if p_obs_end is not None:
        p_o1_raw = p_obs_end
    elif obs_next_pos is not None:
        p_o1_raw = obs_next_pos
    elif obs_velocity is not None and p_o0_raw is not None:
        p_o1_raw = np.asarray(p_o0_raw, dtype=float) + np.asarray(obs_velocity, dtype=float) * float(dt)
    else:
        p_o1_raw = p_o0_raw

    r_o_raw = r_obs if r_obs is not None else (obs_radius if obs_radius is not None else 1.0)

    p_u0 = np.asarray(p_u0_raw, dtype=float)
    p_u1 = np.asarray(p_u1_raw, dtype=float)
    p_o0 = np.asarray(p_o0_raw, dtype=float)
    p_o1 = np.asarray(p_o1_raw, dtype=float)

    r0 = p_u0 - p_o0
    delta_u = p_u1 - p_u0
    delta_o = p_o1 - p_o0
    v_rel = delta_u - delta_o

    v_rel_sq = float(np.dot(v_rel, v_rel))
    collision_threshold = float(r_u_raw + r_o_raw + safety_margin)


    if v_rel_sq < 1e-10:
        # Relative velocity is negligible; distance is constant ||r0||
        dist = float(np.linalg.norm(r0))
        return (dist <= collision_threshold, dist, 0.0)

    # Minimize quadratic D^2(tau) = ||r0||^2 + 2*tau*(r0 . v_rel) + tau^2 * ||v_rel||^2
    tau_opt = -float(np.dot(r0, v_rel)) / v_rel_sq
    tau_min = max(0.0, min(1.0, tau_opt))

    r_min = r0 + tau_min * v_rel
    min_dist = float(np.linalg.norm(r_min))
    has_collided = min_dist <= collision_threshold

    return (has_collided, min_dist, tau_min)


def check_swept_circle_circle(
    p1_start: np.ndarray,
    p1_end: np.ndarray,
    r1: float,
    p2_start: np.ndarray,
    p2_end: np.ndarray,
    r2: float,
) -> bool:
    """Boolean check if two moving circles collide at any point during a timestep."""
    has_collided, _, _ = predict_collision(p1_start, p1_end, r1, p2_start, p2_end, r2)
    return has_collided
